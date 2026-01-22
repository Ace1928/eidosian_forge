from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from ._compatibility import compatibility
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import types
import inspect
import warnings
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair
from .._ops import ops as _ops
def __update_args_kwargs(self, new_args: Tuple['Argument', ...], new_kwargs: Dict[str, 'Argument']):
    """
        This API is internal. Do *not* call it directly.
        """
    self._args = new_args
    self._kwargs = new_kwargs
    for old_use in self._input_nodes.keys():
        old_use.users.pop(self)
    self._input_nodes = {}
    map_arg(self._args, self._input_nodes.setdefault)
    map_arg(self._kwargs, self._input_nodes.setdefault)
    for new_use in self._input_nodes.keys():
        new_use.users.setdefault(self)