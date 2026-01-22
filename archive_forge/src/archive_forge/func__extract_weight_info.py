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
def _extract_weight_info(self, model: GraphModule) -> Dict[str, Dict]:
    """
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the weight information for each layer an observer is attached to.

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping module fqns (str) to a dict with keys:
            "per_channel_max" : maps to the per_channel max values
            "per_channel_min" : maps to the per_channel min values
            "global_max" : maps to the global max recorded
            "global_min" : maps to the global min recorded
        """
    weight_info: Dict[str, Dict] = {}
    for fqn, module in model.named_modules():
        if self._is_supported(module):
            device = module.weight.device
            min_val: torch.Tensor = torch.tensor([float('inf')], device=device)
            max_val: torch.Tensor = torch.tensor([float('-inf')], device=device)
            x_copy = module.weight
            x_dim = x_copy.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x_copy.permute(new_axis_list)
            y = y.to(min_val.dtype)
            y = torch.flatten(y, start_dim=1)
            if min_val.numel() == 0 or max_val.numel() == 0:
                min_val, max_val = torch.aminmax(y, dim=1)
            else:
                min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
                min_val = torch.min(min_val_cur, min_val)
                max_val = torch.max(max_val_cur, max_val)
            weight_info[fqn] = {self.WEIGHT_PREFIX + self.PER_CHANNEL_MAX_KEY: max_val, self.WEIGHT_PREFIX + self.PER_CHANNEL_MIN_KEY: min_val, self.WEIGHT_PREFIX + self.GLOBAL_MAX_KEY: max(max_val), self.WEIGHT_PREFIX + self.GLOBAL_MIN_KEY: min(min_val)}
    return weight_info