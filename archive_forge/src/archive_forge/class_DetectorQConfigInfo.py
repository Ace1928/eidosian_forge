from typing import Any, Dict, Set, Tuple, Callable, List
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.fx._equalize import (
from torch.ao.quantization.observer import _is_activation_post_process
class DetectorQConfigInfo:
    """
    This class contains the QConfig information for a single module.
    The list of variables / values this contains can grow depending on the
    extensibility of the qconfig mapping feature set but this currently includes:
    - if activation observer is dynamic
    - if weight observer is per channel


    Args:
        module_fqn (str): The fully qualified name (fqn) of the module that this
            information contains info relevant to qconfig for
    """

    def __init__(self, module_fqn: str):
        super().__init__()
        self.module_fqn = module_fqn
        self.is_activation_dynamic = False
        self.is_weight_per_channel = False
        self.is_equalization_recommended = False

    def generate_quantization_qconfig(self, module: torch.nn.Module) -> QConfig:
        """
        Args:
            module (torch.nn.Module) The module we are generating
            the qconfig for

        Returns the generated quantization QConfig according to what a valid configuration is
        """
        module_qconfig = default_qconfig
        recommendations_list = []
        recommendations_list.append((self.is_activation_dynamic, self.is_weight_per_channel))
        recommendations_list.append((self.is_activation_dynamic, False))
        recommendations_list.append((False, self.is_weight_per_channel))
        for rec in recommendations_list:
            activation = default_dynamic_quant_observer if rec[0] else default_observer
            weight = default_per_channel_weight_observer if rec[1] else default_weight_observer
            test_config = QConfig(activation, weight)
            try:
                _assert_valid_qconfig(test_config, module)
                module_qconfig = test_config
                break
            except AssertionError:
                continue
        return module_qconfig

    def generate_equalization_qconfig(self) -> EqualizationQConfig:
        """
        This returns the equalization configuration for a module.

        For now, it just returns the default, but as more equalization options become
        possible, this method can get more fleshed out with more nuanced granularity.


        Returns the generated equalization QConfig according to what a valid configuration is
        """
        return default_equalization_qconfig