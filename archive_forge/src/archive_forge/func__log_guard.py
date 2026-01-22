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
def _log_guard(self, prefix: str, g):
    if self.log.isEnabledFor(logging.INFO):
        fsummary, user_tb, maybe_user_loc = self._get_stack_summary()
        is_debug = False
        maybe_extra_debug = ''
        if is_debug and user_tb:
            maybe_extra_debug = '\nUser Stack (most recent call last):\n' + '  (snipped, see stack below for prefix)\n' + ''.join(traceback.format_list(user_tb))
        self.log.info('%s %s [guard added]%s (%s)%s', prefix, g, maybe_user_loc, format_frame(fsummary), maybe_extra_debug, stack_info=is_debug)