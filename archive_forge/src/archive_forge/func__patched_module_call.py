import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def _patched_module_call(self, *args, **kwargs):
    submodule_example_inputs = list(args).copy()
    normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
    num_args = _get_num_pos_args(self.forward) - 1
    num_to_pop = num_args - len(submodule_example_inputs)
    while num_to_pop and normalized_kwargs:
        normalized_kwargs.popitem(last=False)
        num_to_pop -= 1
    submodule_example_inputs.extend(normalized_kwargs.values())
    submodule_example_inputs_tuple = tuple(submodule_example_inputs)
    fqn = _get_path_of_module(root, self)
    if fqn is not None:
        fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
    return orig_module_call(self, *args, **kwargs)