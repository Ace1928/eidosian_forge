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
def _constrain_symbol_range(shape_env, s: sympy.Symbol, compiler_min: int, compiler_max: int, runtime_min: int, runtime_max: int):
    log.debug('_constrain_symbol_range %s [%s, %s] [%s, %s]', s, compiler_min, compiler_max, runtime_min, runtime_max)
    if (r := shape_env.var_to_range.get(s, None)):
        shape_env.var_to_range[s] = ValueRanges(builtins.max(r.lower, compiler_min), builtins.min(r.upper, compiler_max))
    else:
        shape_env.var_to_range[s] = ValueRanges(compiler_min, compiler_max)
    if (r := shape_env.runtime_var_to_range.get(s, None)):
        shape_env.runtime_var_to_range[s] = ValueRanges(builtins.max(r.lower, runtime_min), builtins.min(r.upper, runtime_max))
    else:
        shape_env.runtime_var_to_range[s] = ValueRanges(runtime_min, runtime_max)