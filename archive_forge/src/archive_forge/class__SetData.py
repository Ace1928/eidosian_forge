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
class _SetData(_SetDataBase):
    """The base for all Pyomo AML objects that can be used as a component
    indexing set.

    Derived versions of this class can be used as the Index for any
    IndexedComponent (including IndexedSet)."""
    __slots__ = ()

    def __contains__(self, value):
        try:
            ans = self.get(value, _NotFound)
        except TypeError:
            if isinstance(value, _SetData):
                ans = _NotFound
            else:
                raise
        if ans is _NotFound:
            if isinstance(value, _SetData):
                deprecation_warning("Testing for set subsets with 'a in b' is deprecated.  Use 'a.issubset(b)'.", version='5.7')
                return value.issubset(self)
            else:
                return False
        return True

    def get(self, value, default=None):
        raise DeveloperError('Derived set class (%s) failed to implement get()' % (type(self).__name__,))

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return False

    def isfinite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return False

    def isordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return False

    def subsets(self, expand_all_set_operators=None):
        return iter((self,))

    def __iter__(self):
        """Iterate over the set members

        Raises AttributeError for non-finite sets.  This must be
        declared for non-finite sets because scalar sets inherit from
        IndexedComponent, which provides an iterator (over the
        underlying indexing set).
        """
        raise TypeError("'%s' object is not iterable (non-finite Set '%s' is not iterable)" % (self.__class__.__name__, self.name))

    def __eq__(self, other):
        if self is other:
            return True
        if hasattr(other, 'isfinite'):
            if not other.parent_component().is_constructed():
                return False
            other_isfinite = other.isfinite()
            if not other_isfinite:
                try:
                    other = RangeSet(ranges=list(other.ranges()))
                    other_isfinite = other.isfinite()
                except TypeError:
                    pass
        elif hasattr(other, '__contains__'):
            other_isfinite = True
            try:
                other = set(other)
            except:
                pass
        else:
            return False
        if not self.isfinite():
            try:
                self = RangeSet(ranges=list(self.ranges()))
            except TypeError:
                pass
        if self.isfinite():
            if not other_isfinite:
                return False
            if len(self) != len(other):
                return False
            for x in self:
                if x not in other:
                    return False
            return True
        elif other_isfinite:
            return False
        return self.issubset(other) and other.issubset(self)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        raise DeveloperError('Derived set class (%s) failed to implement __str__' % (type(self).__name__,))

    @property
    def dimen(self):
        raise DeveloperError('Derived set class (%s) failed to implement dimen' % (type(self).__name__,))

    @property
    def domain(self):
        raise DeveloperError('Derived set class (%s) failed to implement domain' % (type(self).__name__,))

    def ranges(self):
        raise DeveloperError('Derived set class (%s) failed to implement ranges' % (type(self).__name__,))

    def bounds(self):
        try:
            _bnds = [(r.start, r.end) if r.step >= 0 else (r.end, r.start) for r in self.ranges()]
        except AttributeError:
            return (None, None)
        if len(_bnds) == 1:
            lb, ub = _bnds[0]
        elif not _bnds:
            return (None, None)
        else:
            lb = min(_bnds, key=itemgetter(0))[0]
            ub = max(_bnds, key=itemgetter(1))[1]
        if lb == -_inf:
            lb = None
        elif int(lb) == lb:
            lb = int(lb)
        if ub == _inf:
            ub = None
        elif int(ub) == ub:
            ub = int(ub)
        return (lb, ub)

    def get_interval(self):
        """Return the interval for this Set as (start, end, step)

        Returns the effective interval for this Set as a (start, end,
        step) tuple.  Start and End are the same as returned by
        `bounds()`.  Step is 0 for continuous ranges, a positive value
        for regular discrete sets (e.g., 1 for Integers), or `None` for
        Sets that do not have a regular interval (e.g., semicontinuous
        sets, mixed type sets, sets with dimen != 1, etc).

        """
        if self.dimen != 1:
            return self.bounds() + (None,)
        if self.isdiscrete():
            return self._get_discrete_interval()
        else:
            return self._get_continuous_interval()

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

    def _get_continuous_interval(self):
        ranges = []
        discrete = []
        for r in self.ranges():
            if r.isdiscrete():
                discrete.append(r)
            else:
                ranges.append(NumericRange(r.start, r.end, r.step, r.closed))
        if len(ranges) == 1 and (not discrete):
            r = ranges[0]
            return (None if r.start == -_inf else r.start, None if r.end == _inf else r.end, abs(r.step))
        for r in ranges:
            if not r.closed[0]:
                for d in discrete:
                    if r.start in d:
                        r.closed = (True, r.closed[1])
                        break
            if not r.closed[1]:
                for d in discrete:
                    if r.end in d:
                        r.closed = (r.closed[0], True)
                        break
        nRanges = len(ranges)
        r = ranges.pop()
        interval = NumericRange(r.start, r.end, r.step, r.closed)
        _rlen = len(ranges)
        while _rlen and nRanges > _rlen:
            nRanges = _rlen
            for i, r in enumerate(ranges):
                if interval.isdisjoint(r):
                    continue
                ranges[i] = None
                if r.start < interval.start:
                    interval.start = r.start
                    interval.closed = (r.closed[0], interval.closed[1])
                elif not interval.closed[0] and r.start == interval.start:
                    interval.closed = (r.closed[0], interval.closed[1])
                if r.end > interval.end:
                    interval.end = r.end
                    interval.closed = (interval.closed[0], r.closed[1])
                elif not interval.closed[1] and r.end == interval.end:
                    interval.closed = (interval.closed[0], r.closed[1])
            ranges = list((_ for _ in ranges if _ is not None))
            _rlen = len(ranges)
        if ranges:
            return self.bounds() + (None,)
        for r in discrete:
            if not r.issubset(interval):
                return self.bounds() + (None,)
        start = interval.start
        if start == -_inf:
            start = None
        end = interval.end
        if end == _inf:
            end = None
        return (start, end, interval.step)

    @property
    @deprecated("The 'virtual' attribute is no longer supported", version='5.7')
    def virtual(self):
        return isinstance(self, (_AnySet, SetOperator, _InfiniteRangeSetData))

    @virtual.setter
    def virtual(self, value):
        if value != self.virtual:
            raise ValueError("Attempting to set the (deprecated) 'virtual' attribute on %s to an invalid value (%s)" % (self.name, value))

    @property
    @deprecated("The 'concrete' attribute is no longer supported.  Use isdiscrete() or isfinite()", version='5.7')
    def concrete(self):
        return self.isfinite()

    @concrete.setter
    def concrete(self, value):
        if value != self.concrete:
            raise ValueError("Attempting to set the (deprecated) 'concrete' attribute on %s to an invalid value (%s)" % (self.name, value))

    @property
    @deprecated("The 'ordered' attribute is no longer supported.  Use isordered()", version='5.7')
    def ordered(self):
        return self.isordered()

    @property
    @deprecated("'filter' is no longer a public attribute.", version='5.7')
    def filter(self):
        return None

    @deprecated('check_values() is deprecated: Sets only contain valid members', version='5.7')
    def check_values(self):
        """
        Verify that the values in this set are valid.
        """
        return True

    def isdisjoint(self, other):
        """Test if this Set is disjoint from `other`

        Parameters
        ----------
            other : ``Set`` or ``iterable``
                The Set or iterable object to compare this Set against

        Returns
        -------
        bool : True if this set is disjoint from `other`
        """
        if hasattr(other, 'isfinite'):
            other_isfinite = other.isfinite()
        elif hasattr(other, '__contains__'):
            other_isfinite = True
            try:
                other = set(other)
            except:
                pass
        else:
            raise TypeError("'%s' object is not iterable" % (type(other).__name__,))
        if self.isfinite():
            for x in self:
                if x in other:
                    return False
            return True
        elif other_isfinite:
            for x in other:
                if x in self:
                    return False
            return True
        else:
            all((r.isdisjoint(s) for r in self.ranges() for s in other.ranges()))

    def issubset(self, other):
        """Test if this Set is a subset of `other`

        Parameters
        ----------
            other : ``Set`` or ``iterable``
                The Set or iterable object to compare this Set against

        Returns
        -------
        bool : True if this set is a subset of `other`
        """
        if hasattr(other, 'isfinite'):
            other_isfinite = other.isfinite()
            if not other_isfinite:
                try:
                    other = RangeSet(ranges=list(other.ranges()))
                    other_isfinite = other.isfinite()
                except TypeError:
                    pass
        elif hasattr(other, '__contains__'):
            other_isfinite = True
            try:
                other = set(other)
            except:
                pass
        else:
            raise TypeError("'%s' object is not iterable" % (type(other).__name__,))
        if not self.isfinite():
            try:
                self = RangeSet(ranges=list(self.ranges()))
            except TypeError:
                pass
        if self.isfinite():
            for x in self:
                if x not in other:
                    return False
            return True
        elif other_isfinite:
            return False
        else:
            for r in self.ranges():
                try:
                    if r.range_difference(other.ranges()):
                        return False
                except RangeDifferenceError:
                    return False
            return True

    def issuperset(self, other):
        """Test if this Set is a superset of `other`

        Parameters
        ----------
            other : ``Set`` or ``iterable``
                The Set or iterable object to compare this Set against

        Returns
        -------
        bool : True if this set is a superset of `other`
        """
        if hasattr(other, 'isfinite'):
            other_isfinite = other.isfinite()
            if not other_isfinite:
                try:
                    other = RangeSet(ranges=list(other.ranges()))
                    other_isfinite = other.isfinite()
                except TypeError:
                    pass
        elif hasattr(other, '__contains__'):
            other_isfinite = True
            try:
                other = set(other)
            except:
                pass
        else:
            raise TypeError("'%s' object is not iterable" % (type(other).__name__,))
        if other_isfinite:
            for x in other:
                try:
                    if x not in self:
                        return False
                except TypeError:
                    return False
            return True
        if not self.isfinite():
            try:
                self = RangeSet(ranges=list(self.ranges()))
            except TypeError:
                pass
        if self.isfinite():
            return False
        else:
            return other.issubset(self)

    def union(self, *args):
        """
        Return the union of this set with one or more sets.
        """
        tmp = self
        for arg in args:
            tmp = SetUnion(tmp, arg)
        return tmp

    def intersection(self, *args):
        """
        Return the intersection of this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = SetIntersection(tmp, arg)
        return tmp

    def difference(self, *args):
        """
        Return the difference between this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = SetDifference(tmp, arg)
        return tmp

    def symmetric_difference(self, other):
        """
        Return the symmetric difference of this set with another set
        """
        return SetSymmetricDifference(self, other)

    def cross(self, *args):
        """
        Return the cross-product between this set and one or more sets
        """
        return SetProduct(self, *args)
    __le__ = issubset
    __ge__ = issuperset
    __or__ = union
    __and__ = intersection
    __sub__ = difference
    __xor__ = symmetric_difference
    __mul__ = cross

    def __ror__(self, other):
        return SetUnion(other, self)

    def __rand__(self, other):
        return SetIntersection(other, self)

    def __rsub__(self, other):
        return SetDifference(other, self)

    def __rxor__(self, other):
        return SetSymmetricDifference(other, self)

    def __rmul__(self, other):
        return SetProduct(other, self)

    def __lt__(self, other):
        """
        Return True if the set is a strict subset of 'other'
        """
        return self <= other and (not self == other)

    def __gt__(self, other):
        """
        Return True if the set is a strict superset of 'other'
        """
        return self >= other and (not self == other)