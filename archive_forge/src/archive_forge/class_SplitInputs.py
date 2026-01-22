from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
class SplitInputs:
    all_tensors: List[Any]
    tensor_args: List[Any]
    nontensor_args: List[Any]
    arg_types: List[str]
    tensor_kwargs: Dict[str, Any]
    kwarg_order: List[str]
    nontensor_kwargs: Dict[str, Any]
    kwarg_types: Dict[str, Any]

    @staticmethod
    def _is_tensor_input(arg):
        return isinstance(arg, torch.Tensor) or is_iterable_of_tensors(arg)

    def __init__(self, args, kwargs):
        self.arg_types = ['t' if self._is_tensor_input(arg) else 's' for arg in args]
        self.kwarg_types = {k: 't' if self._is_tensor_input(v) else 's' for k, v in kwargs.items()}
        self.tensor_args = [arg for arg in args if self._is_tensor_input(arg)]
        self.nontensor_args = [arg for arg in args if not self._is_tensor_input(arg)]
        self.tensor_kwargs = {k: v for k, v in kwargs.items() if self._is_tensor_input(v)}
        self.nontensor_kwargs = {k: v for k, v in kwargs.items() if not self._is_tensor_input(v)}
        self.all_tensors = [*self.tensor_args, *[v for k, v in self.tensor_kwargs.items()]]
        self.kwarg_order = [k for k, v in kwargs.items()]

    def nontensors_match(self, other: 'SplitInputs'):
        if self.arg_types != other.arg_types:
            return False
        if self.kwarg_types != other.kwarg_types:
            return False
        if self.kwarg_order != other.kwarg_order:
            return False
        if self.nontensor_args != other.nontensor_args:
            return False
        if self.nontensor_kwargs != other.nontensor_kwargs:
            return False
        return True