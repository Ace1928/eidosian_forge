import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
class OrtExecutionInfoPerSession:
    """Information required to execute torch.fx.GraphModule using onnxruntime.InferenceSession"""

    def __init__(self, session: 'onnxruntime.InferenceSession', input_names: Tuple[str, ...], input_value_infos: Tuple['onnx.ValueInfoProto', ...], output_names: Tuple[str, ...], output_value_infos: Tuple['onnx.ValueInfoProto', ...], input_devices: Tuple['ORTC.OrtDevice', ...], output_devices: Tuple['ORTC.OrtDevice', ...], example_outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):
        self.session: onnxruntime.InferenceSession = session
        self.input_names: Tuple[str, ...] = input_names
        self.input_value_infos: Tuple[onnx.ValueInfoProto, ...] = input_value_infos
        self.output_names: Tuple[str, ...] = output_names
        self.output_value_infos: Tuple[onnx.ValueInfoProto, ...] = output_value_infos
        self.input_devices: Tuple['ORTC.OrtDevice', ...] = input_devices
        self.output_devices: Tuple['ORTC.OrtDevice', ...] = output_devices
        self.example_outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor] = example_outputs

    def is_supported(self, *args):
        if len(args) != len(self.input_value_infos):
            return False
        for arg, value_info in zip(args, self.input_value_infos):
            if not isinstance(arg, torch.Tensor):
                return False
            onnx_dtype = _TORCH_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE[arg.dtype]
            if onnx_dtype != value_info.type.tensor_type.elem_type:
                return False
            for dim, onnx_dim in zip(arg.shape, value_info.type.tensor_type.shape.dim):
                if isinstance(dim, int) and (onnx_dim.dim_value == dim or onnx_dim.dim_param):
                    continue
                elif isinstance(dim, torch.SymInt) and onnx_dim.dim_param:
                    continue
                else:
                    return False
        return True