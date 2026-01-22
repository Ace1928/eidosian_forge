import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
@record_shapeenv_event()
def _constrain_range_for_size(a, min: Optional[int]=None, max: Optional[int]=None):
    """
    This function is NOT INTENDED to be used by itself.
    """
    if isinstance(a, (SymFloat, SymBool)):
        raise ValueError('Constraining SymFloat/SymBool is nyi')
    assert isinstance(a, SymInt), 'can only constrain range for SymInt'
    assert isinstance(a.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
    if min is None:
        min = 0
    if max is None:
        max = sympy.oo
    if max <= 2:
        raise ValueError(f'Maximum value to constrain_as_size must be greater than 2, but was {max}')
    if max < min:
        raise ValueError("Maximum value to constrain_as_size can't be less than the specified min value, received min={min} and max={max}")
    compiler_min = 2 if min < 2 else min
    _constrain_symbol_range(a.node.shape_env, a.node.expr, compiler_min=compiler_min, compiler_max=max, runtime_min=min, runtime_max=max)