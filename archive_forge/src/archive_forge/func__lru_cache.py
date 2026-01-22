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
def _lru_cache(fn, maxsize=None):
    """
    Wrapper around lru_cache that clears when new info about shapes has been
    updated.

    Use lru_cache if the output is always the same, regardless of the
    constraints we know now (i.e. evaluate_expr)

    Use _lru_cache otherwise.

    Also note that this depends on _update_version_counter being called on the
    shape environment whenever the constraints are updated, otherwise the cache
    will not be cleared.
    """
    fn_cache = lru_cache(maxsize)(fn)
    prior_version = 0
    if config.validate_shape_env_verison_key:
        prior_key = None

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            nonlocal prior_version, prior_key
            if prior_key is None:
                prior_key = self._get_key()
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
                prior_key = self._get_key()
            else:
                assert prior_key == self._get_key(), 'ShapeEnv cache key changed without version being updated!'
            return fn_cache(self, *args, **kwargs)
    else:

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            nonlocal prior_version
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
            return fn_cache(self, *args, **kwargs)
    wrapper.cache_clear = fn_cache.cache_clear
    wrapper.cache_info = fn_cache.cache_info
    return wrapper