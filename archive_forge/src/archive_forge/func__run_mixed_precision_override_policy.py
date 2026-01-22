import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def _run_mixed_precision_override_policy(root_module: nn.Module, module_classes: Iterable[Type[nn.Module]], ignored_modules: Set[nn.Module], root_kwargs: Dict[str, Any], target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]]):
    module_classes_tuple = tuple(set(module_classes))
    for module in root_module.modules():
        if module in ignored_modules:
            continue
        elif isinstance(module, module_classes_tuple):
            if module not in target_module_to_kwargs:
                target_module_to_kwargs[module] = root_kwargs
            target_module_to_kwargs[module]['mixed_precision'] = None
    return target_module_to_kwargs