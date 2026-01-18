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
def track_symint(source, val, constraint=None):
    log.debug('track_symint %s %s %s', LazyString(source.name), val, constraint)
    assert not isinstance(val, SymInt) or is_symbolic(val)
    if isinstance(val, SymInt) and val.node.maybe_as_int() is not None:
        val = val.node.maybe_as_int()
    if isinstance(val, SymInt):
        s = val.node.expr
        if isinstance(s, sympy.Symbol):
            symbol_to_source[s].append(source)
            if constraint is not None:
                symbol_to_constraints[s].add(constraint)
        elif isinstance(-s, sympy.Symbol):
            symbol_to_source[-s].append(NegateSource(source))
        else:
            constraint_violated = False
            if isinstance(constraint, StrictMinMaxConstraint):
                sym_vrs = {x: self.var_to_range.get(x, None) for x in s.free_symbols}
                if all((vr is not None for vr in sym_vrs.values())):
                    expr_vr = bound_sympy(s, sym_vrs)
                    if expr_vr != constraint.vr:
                        constraint_violated = True
                else:
                    constraint_violated = True
            elif isinstance(constraint, RelaxedUnspecConstraint):
                if s.is_number:
                    i = int(s)
                    if i not in (0, 1):
                        constraint_violated = True
                else:
                    constraint_violated = True
            if constraint_violated:

                def hint(s):
                    sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(s)
                    return f'{sexpr}.'
                var_with_range = self.render_range_for_constraint_violation(source, constraint)
                msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be equal to '
                record_constraint_violation(constraint.warn_only, self.debug_name(source), msg, hint=functools.partial(hint, s))
        input_guards.append((source, s))
    else:
        s = sympy.Integer(val)
        input_guards.append((source, s))
        constraint_violated = False
        if isinstance(constraint, StrictMinMaxConstraint):
            constraint_violated = True
        elif isinstance(constraint, RelaxedUnspecConstraint):
            if val not in (0, 1):
                constraint_violated = True
        if constraint_violated:
            var_with_range = self.render_range_for_constraint_violation(source, constraint)
            msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be a constant ({val}).'
            record_constraint_violation(constraint.warn_only, self.debug_name(source), msg)