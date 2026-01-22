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
def _pretty_print_target(self, target):
    """
        Make target printouts more user-friendly.
        1) builtins will be printed as `builtins.xyz`
        2) operators will be printed as `operator.xyz`
        3) other callables will be printed with qualified name, e.g. torch.add
        """
    if isinstance(target, str):
        return target
    if hasattr(target, '__module__'):
        if not hasattr(target, '__name__'):
            return _get_qualified_name(target)
        if target.__module__ == 'builtins':
            return f'builtins.{target.__name__}'
        elif target.__module__ == '_operator':
            return f'operator.{target.__name__}'
    return _get_qualified_name(target)