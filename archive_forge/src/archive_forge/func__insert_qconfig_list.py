from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Union
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER
from torch.ao.quantization.qconfig import QConfigAny
def _insert_qconfig_list(self, style: str, args: List[Union[str, int, Callable]], qconfig_list: List[QConfigAny]) -> None:
    _remove_duplicates_and_none(qconfig_list)
    self._handle_list_size_mismatch(qconfig_list, style)
    method_name = _QCONFIG_STYLE_TO_METHOD[style]
    for qconfig_mapping, qconfig in zip(self.qconfig_mappings_list, qconfig_list):
        set_method = getattr(qconfig_mapping, method_name)
        set_method(*args, qconfig)