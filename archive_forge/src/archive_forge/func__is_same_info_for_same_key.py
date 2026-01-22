from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def _is_same_info_for_same_key(self, info_dict_a: Dict, info_dict_b: Dict) -> bool:
    """
        Takes in two dictionaries and ensures that any common keys between the two have the same
        values.

        Args:
            info_dict_a (Dict): First dictionary we wish to compare
            info_dict_b (Dict): Second dictionary we wish to compare

        Returns True if all shared keys have same values, false otherwise
        """
    dict_a_keys: Set = set(info_dict_a.keys())
    dict_b_keys: Set = set(info_dict_b.keys())
    intersecting_keys: Set = dict_a_keys.intersection(dict_b_keys)
    for key in intersecting_keys:
        dict_a_val = info_dict_a[key]
        dict_b_val = info_dict_b[key]
        if type(dict_a_val) == torch.Tensor:
            if type(dict_b_val) != torch.Tensor or sum(dict_a_val != dict_b_val) != 0:
                return False
        elif dict_a_val != dict_b_val:
            return False
    return True