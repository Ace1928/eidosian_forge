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
def _generate_comparison_values(self, input_info: Dict, weight_info: Dict) -> Dict[str, torch.Tensor]:
    """
        Takes in the information on the min and max values of the inputs and weights and:
            Calculates the comp stat for each channel: s_c = sqrt(w_c/W)/sqrt(i_c/I)

        Args:
            input_info (dict): A dict mapping each observer to input range information
            weight_info (dict): A dict mapping each observer to weight range information

        Returns a dict mapping relevant observer fqns (str) to a 1-D tensor.
            Each value is a different s_c value for a different channel
        """
    module_fqn_to_channel: Dict[str, torch.Tensor] = {}
    for module_fqn in input_info:
        if module_fqn not in weight_info:
            raise KeyError(f'Unable to find weight range stats for module {module_fqn}')
        weight_ratio = self._calculate_range_ratio(weight_info[module_fqn], self.WEIGHT_STR, module_fqn)
        input_ratio = self._calculate_range_ratio(input_info[module_fqn], self.INPUT_STR, module_fqn)
        weight_channels = len(weight_ratio)
        input_channels = len(input_ratio)
        if weight_channels != input_channels:
            assert input_channels % weight_channels == 0, 'input channels should be divisible by weight channels.'
            rep_factor: int = input_channels // weight_channels
            weight_ratio = weight_ratio.repeat(rep_factor)
        s = torch.sqrt(weight_ratio) / torch.sqrt(input_ratio)
        module_fqn_to_channel[module_fqn] = s
    return module_fqn_to_channel