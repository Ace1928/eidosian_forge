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
@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """
    Given pairs of sources corresponding to pairs of dynamic dimensions that
    are specified equal, represent them in a union-find data structure so that
    we can efficiently check whether two such sources are transitively equal.
    """
    source_pairs: List[Tuple[Source, Source]]

    def __post_init__(self):
        object.__setattr__(self, '_parents', {})
        for source1, source2 in self.source_pairs:
            self._union(self._find(source1), self._find(source2))

    def _find(self, source):
        if source in self._parents:
            return self._find(self._parents[source])
        else:
            return source

    def _union(self, root1, root2):
        if root1 != root2:
            self._parents[root1] = root2

    def render(self):
        buf = ', '.join((f'{source1.name()} == {source2.name()}' for source1, source2 in self.source_pairs))
        return '{' + buf + '}'

    def is_equal(self, source1, source2):
        return self._find(source1) == self._find(source2)