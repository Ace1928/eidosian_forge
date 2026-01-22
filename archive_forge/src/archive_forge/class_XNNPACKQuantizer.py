from __future__ import annotations
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
class XNNPACKQuantizer(Quantizer):
    supported_config_and_operators = _get_supported_config_and_operators()
    STATIC_QAT_ONLY_OPS = ['conv_bn_relu', 'conv_bn']
    STATIC_OPS = ['linear', 'conv_relu', 'conv', 'adaptive_avg_pool2d', 'gru_io_only', 'max_pool2d', 'add_relu', 'add', 'mul_relu', 'mul', 'cat']
    DYNAMIC_OPS = ['linear']

    def __init__(self):
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.operator_type_config: Dict[torch._ops.OpOverloadPacket, Optional[QuantizationConfig]] = {}
        self.module_type_config: Dict[Callable, Optional[QuantizationConfig]] = {}
        self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.supported_config_and_operators:
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(cls, quantization_config: Optional[QuantizationConfig]) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops
        for config, ops in cls.supported_config_and_operators:
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: QuantizationConfig) -> XNNPACKQuantizer:
        self.global_config = quantization_config
        return self

    def set_operator_type(self, operator_type: torch._ops.OpOverloadPacket, quantization_config: QuantizationConfig) -> XNNPACKQuantizer:
        self.operator_type_config[operator_type] = quantization_config
        return self

    def set_module_type(self, module_type: Callable, quantization_config: QuantizationConfig):
        """Set quantization_config for a submodule with type: `module_type`, for example:
        quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
        patterns in the submodule with this module type with the given `quantization_config`
        """
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(self, module_name: str, quantization_config: Optional[QuantizationConfig]):
        """Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`
        """
        assert quantization_config is not None, ' quantization_config == None is not supported yet'
        self.module_name_config[module_name] = quantization_config
        return self

    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
        return _convert_scalars_to_attrs(model)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        if self.global_config and self.global_config.input_activation.is_dynamic:
            model = self._annotate_for_dynamic_quantization_config(model)
        else:
            model = self._annotate_for_static_quantization_config(model)
        propagate_annotation(model)
        return model

    def _annotate_all_static_patterns(self, model: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> torch.fx.GraphModule:
        if quantization_config is None:
            return model
        if quantization_config.is_qat:
            for op in self.STATIC_QAT_ONLY_OPS:
                OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        for op in self.STATIC_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        return model

    def _annotate_all_dynamic_patterns(self, model: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> torch.fx.GraphModule:
        if quantization_config is None:
            return model
        for op in self.DYNAMIC_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        return model

    def _annotate_for_static_quantization_config(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(model, config, _get_module_name_filter(module_name))
        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_all_static_patterns(model, config, _get_module_type_filter(module_type))
        self._annotate_all_static_patterns(model, self.global_config, _get_not_module_type_or_name_filter(tp_list, module_name_list))
        return model

    def _annotate_for_dynamic_quantization_config(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_dynamic_patterns(model, config, _get_module_name_filter(module_name))
        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_all_dynamic_patterns(model, config, _get_module_type_filter(module_type))
        self._annotate_all_dynamic_patterns(model, self.global_config, _get_not_module_type_or_name_filter(tp_list, module_name_list))
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators