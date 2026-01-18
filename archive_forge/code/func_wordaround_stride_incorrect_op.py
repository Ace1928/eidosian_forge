import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
@register_op_impl(stride_incorrect_op)
def wordaround_stride_incorrect_op(fake_mode, func, *args, **kwargs):

    def is_symbolic(x):
        if isinstance(x, FakeTensor):
            return x._has_symbolic_sizes_strides
        if isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return True
        return False
    if fake_mode.allow_fallback_kernels:
        require_dynamic = any((is_symbolic(x) for x in itertools.chain(args, kwargs.values())))
        if not require_dynamic:
            flat_args, args_spec = pytree.tree_flatten((args, kwargs))
            return run_fallback_kernel(fake_mode, func, flat_args, args_spec, None)
    raise UnsupportedOperatorException(func)