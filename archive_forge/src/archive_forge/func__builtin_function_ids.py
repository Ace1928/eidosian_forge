import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
@FunctionIdSet
def _builtin_function_ids() -> Dict[int, str]:
    rv = {id(v): f'builtins.{k}' for k, v in builtins.__dict__.items() if not k.startswith('_') and callable(v)}
    rv.update({id(v): f'operator.{k}' for k, v in operator.__dict__.items() if not k.startswith('_') and callable(v)})
    rv.update({id(v): f'functools.{v.__name__}' for v in (itertools.chain, itertools.islice)})
    rv.update({id(cast): 'typing.cast', id(functools.reduce): 'functools.reduce', id(copy.deepcopy): 'copy.deepcopy'})
    return rv