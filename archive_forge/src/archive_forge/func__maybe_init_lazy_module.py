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
def _maybe_init_lazy_module(obj: object) -> None:
    module = getattr(obj, '__module__', None)
    if module is None:
        return
    base_module = module.split('.')[0]
    init_funcs = _lazy_module_init.pop(base_module, None)
    if init_funcs is not None:
        for fn in init_funcs:
            fn()