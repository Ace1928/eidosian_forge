import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
def _get_discrete_interval(self):
    ranges = list(self.ranges())
    if len(ranges) == 1:
        try:
            start, end, c = ranges[0].normalize_bounds()
        except AttributeError:
            return self.bounds() + (None,)
        return (None if start == -_inf else start, None if end == _inf else end, abs(ranges[0].step))
    try:
        step = min((abs(r.step) for r in ranges if r.step != 0))
    except ValueError:
        vals = sorted(self)
        if len(vals) < 2:
            return (vals[0], vals[0], 0)
        step = vals[1] - vals[0]
        for i in range(2, len(vals)):
            if step != vals[i] - vals[i - 1]:
                return self.bounds() + (None,)
        return (vals[0], vals[-1], step)
    except AttributeError:
        return self.bounds() + (None,)
    nRanges = len(ranges)
    r = ranges.pop()
    _rlen = len(ranges)
    ref = r.start
    if r.step >= 0:
        start, end = (r.start, r.end)
    else:
        end, start = (r.start, r.end)
    if r.step % step:
        return self.bounds() + (None,)
    for r in ranges:
        if (r.start - ref) % step:
            return self.bounds() + (None,)
        if r.step % step:
            return self.bounds() + (None,)
    while nRanges > _rlen:
        nRanges = _rlen
        for i, r in enumerate(ranges):
            if r.step > 0:
                rstart, rend = (r.start, r.end)
            else:
                rend, rstart = (r.start, r.end)
            if not r.step or abs(r.step) == step:
                if start <= rend + step and rstart <= end + step:
                    ranges[i] = None
                    if start > rstart:
                        start = rstart
                    if end < rend:
                        end = rend
            elif start <= rstart + step and end >= rend - step:
                ranges[i] = None
                if start > rstart:
                    start = rstart
                if end < rend:
                    end = rend
        ranges = list((_ for _ in ranges if _ is not None))
        _rlen = len(ranges)
    if ranges:
        return self.bounds() + (None,)
    return (None if start == -_inf else start, None if end == _inf else end, step)