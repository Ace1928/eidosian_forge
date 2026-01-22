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
class _FiniteRangeSetData(_SortedSetMixin, _OrderedSetMixin, _FiniteSetMixin, _InfiniteRangeSetData):
    __slots__ = ()

    @staticmethod
    def _range_gen(r):
        start, end = (r.start, r.end) if r.step > 0 else (r.end, r.start)
        step = abs(r.step)
        n = start
        i = 0
        if start == end:
            yield start
        else:
            while n <= end:
                yield n
                i += 1
                n = start + i * step

    def _iter_impl(self):
        nIters = len(self._ranges) - 1
        if not nIters:
            yield from _FiniteRangeSetData._range_gen(self._ranges[0])
            return
        iters = []
        for r in self._ranges:
            i = _FiniteRangeSetData._range_gen(r)
            iters.append([next(i), i])
        iters.sort(reverse=True, key=lambda x: x[0])
        n = None
        while iters:
            if n != iters[-1][0]:
                n = iters[-1][0]
                yield n
            try:
                iters[-1][0] = next(iters[-1][1])
                if nIters and iters[-2][0] < iters[-1][0]:
                    iters.sort(reverse=True)
            except StopIteration:
                iters.pop()
                nIters -= 1

    def __len__(self):
        if len(self._ranges) == 1:
            r = self._ranges[0]
            if r.start == r.end:
                return 1
            else:
                return int((r.end - r.start) // r.step) + 1
        else:
            return sum((1 for _ in self))

    def at(self, index):
        assert int(index) == index
        idx = self._to_0_based_index(index)
        if len(self._ranges) == 1:
            r = self._ranges[0]
            ans = r.start + idx * r.step
            if ans <= r.end:
                return ans
        else:
            for ans in self:
                if not idx:
                    return ans
                idx -= 1
        raise IndexError(f'{self.name} index out of range')

    def ord(self, item):
        if len(self._ranges) == 1:
            r = self._ranges[0]
            i = float(item - r.start) / r.step
            if item >= r.start and item <= r.end and (abs(i - math.floor(i + 0.5)) < r._EPS):
                return int(math.floor(i + 0.5)) + 1
        else:
            ans = 1
            for val in self:
                if val == item:
                    return ans
                ans += 1
        raise ValueError('Cannot identify position of %s in Set %s: item not in Set' % (item, self.name))
    bounds = _InfiniteRangeSetData.bounds
    ranges = _InfiniteRangeSetData.ranges
    domain = _InfiniteRangeSetData.domain