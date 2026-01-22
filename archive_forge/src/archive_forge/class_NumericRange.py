import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
class NumericRange(object):
    """A representation of a numeric range.

    This class represents a contiguous range of numbers.  The class
    mimics the Pyomo (*not* Python) `range` API, with a Start, End, and
    Step.  The Step is a signed int.  If the Step is 0, the range is
    continuous.  The End *is* included in the range.  Ranges are closed,
    unless `closed` is specified as a 2-tuple of bool values.  Only
    continuous ranges may be open (or partially open)

    Closed ranges are not necessarily strictly finite, as None is
    allowed for the End value (as well as the Start value, for
    continuous ranges only).

    Parameters
    ----------
        start : float
            The starting value for this NumericRange
        end : float
            The last value for this NumericRange
        step : int
            The interval between values in the range.  0 indicates a
            continuous range.  Negative values indicate discrete ranges
            walking backwards.
        closed : tuple of bool, optional
            A 2-tuple of bool values indicating if the beginning and end
            of the range is closed.  Open ranges are only allowed for
            continuous NumericRange objects.
    """
    __slots__ = ('start', 'end', 'step', 'closed')
    _EPS = 1e-15
    _types_comparable_to_int = {int}
    _closedMap = {True: True, False: False, '[': True, ']': True, '(': False, ')': False}

    def __init__(self, start, end, step, closed=(True, True)):
        if int(step) != step:
            raise ValueError('NumericRange step must be int (got %s)' % (step,))
        step = int(step)
        if start is None:
            start = -_inf
        if end is None:
            end = math.copysign(_inf, step)
        if step:
            if start == -_inf:
                raise ValueError('NumericRange: start must not be None/-inf for non-continuous steps')
            if (end - start) * step < 0:
                raise ValueError('NumericRange: start, end ordering incompatible with step direction (got [%s:%s:%s])' % (start, end, step))
            if end not in _infinite:
                n = int((end - start) // step)
                new_end = start + n * step
                assert abs(end - new_end) < abs(step)
                end = new_end
                if step < 0:
                    start, end = (end, start)
                    step *= -1
        elif end < start:
            raise ValueError('NumericRange: start must be <= end for continuous ranges (got %s..%s)' % (start, end))
        if start == end:
            step = 0
        self.start = start
        self.end = end
        self.step = step
        self.closed = (self._closedMap[closed[0]], self._closedMap[closed[1]])
        if self.isdiscrete() and self.closed != (True, True):
            raise ValueError('NumericRange %s is discrete, but passed closed=%s.  Discrete ranges must be closed.' % (self, self.closed))

    def __getstate__(self):
        """
        Retrieve the state of this object as a dictionary.

        This method must be defined because this class uses slots.
        """
        state = {}
        for i in NumericRange.__slots__:
            state[i] = getattr(self, i)
        return state

    def __setstate__(self, state):
        """
        Set the state of this object using values from a state dictionary.

        This method must be defined because this class uses slots.
        """
        for key, val in state.items():
            object.__setattr__(self, key, val)

    def __str__(self):
        if not self.isdiscrete():
            return '%s%s..%s%s' % ('[' if self.closed[0] else '(', self.start, self.end, ']' if self.closed[1] else ')')
        if self.start == self.end:
            return '[%s]' % (self.start,)
        elif self.step == 1:
            return '[%s:%s]' % (self.start, self.end)
        else:
            return '[%s:%s:%s]' % (self.start, self.end, self.step)
    __repr__ = __str__

    def __eq__(self, other):
        if type(other) is not NumericRange:
            return False
        return self.start == other.start and self.end == other.end and (self.step == other.step) and (self.closed == other.closed)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        if value.__class__ not in self._types_comparable_to_int:
            if check_if_numeric_type(value):
                self._types_comparable_to_int.add(value.__class__)
            else:
                try:
                    if hasattr(value, '__len__') and hasattr(value, '__getitem__') and (len(value) == 1) and (value[0] is not value):
                        return value[0] in self
                except:
                    pass
                try:
                    if not bool(value - 0 > 0) ^ bool(value - 0 <= 0):
                        return False
                    elif value.__class__(0) != 0 or not value.__class__(0) == 0:
                        return False
                    else:
                        self._types_comparable_to_int.add(value.__class__)
                except:
                    return False
        if self.step:
            _dir = int(math.copysign(1, self.step))
            _from_start = value - self.start
            return 0 <= _dir * _from_start <= _dir * (self.end - self.start) and abs(remainder(_from_start, self.step)) <= self._EPS
        else:
            return (value >= self.start if self.closed[0] else value > self.start) and (value <= self.end if self.closed[1] else value < self.end)

    @staticmethod
    def _continuous_discrete_disjoint(cont, disc):
        d_lb = disc.start if disc.step > 0 else disc.end
        d_ub = disc.end if disc.step > 0 else disc.start
        if cont.start <= d_lb:
            return False
        if cont.end >= d_ub:
            return False
        EPS = NumericRange._EPS
        if cont.end - cont.start - EPS > abs(disc.step):
            return False
        rStart = remainder(cont.start - disc.start, abs(disc.step))
        rEnd = remainder(cont.end - disc.start, abs(disc.step))
        return (abs(rStart) > EPS or not cont.closed[0]) and (abs(rEnd) > EPS or not cont.closed[1]) and (rStart - rEnd > 0 or not any(cont.closed))

    def isdiscrete(self):
        return self.step or self.start == self.end

    def isfinite(self):
        return self.step and self.end not in _infinite or self.end == self.start

    def isdisjoint(self, other):
        if not isinstance(other, NumericRange):
            return other.isdisjoint(self)
        if self._nooverlap(other):
            return True
        if not self.step or not other.step:
            if self.start == self.end:
                return self.start not in other
            elif other.start == other.end:
                return other.start not in self
            if self.step:
                return NumericRange._continuous_discrete_disjoint(other, self)
            elif other.step:
                return NumericRange._continuous_discrete_disjoint(self, other)
            else:
                return False
        if self.step == other.step:
            return abs(remainder(other.start - self.start, self.step)) > self._EPS
        elif self.end in _infinite and other.end in _infinite and (self.step * other.step > 0):
            gcd = NumericRange._gcd(self.step, other.step)
            return abs(remainder(other.start - self.start, gcd)) > self._EPS
        if self.step > 0:
            end = min(self.end, max(other.start, other.end))
        else:
            end = max(self.end, min(other.start, other.end))
        i = 0
        item = self.start
        while self.step > 0 and item <= end or (self.step < 0 and item >= end):
            if item in other:
                return False
            i += 1
            item = self.start + self.step * i
        return True

    def issubset(self, other):
        if not isinstance(other, NumericRange):
            if type(other) is AnyRange:
                return True
            elif type(other) is NonNumericRange:
                return False
        s1, e1, c1 = self.normalize_bounds()
        s2, e2, c2 = other.normalize_bounds()
        if s1 < s2:
            return False
        if e1 > e2:
            return False
        if s1 == s2 and c1[0] and (not c2[0]):
            return False
        if e1 == e2 and c1[1] and (not c2[1]):
            return False
        if other.step == 0:
            return True
        elif self.step == 0:
            if self.start == self.end:
                return self.start in other
            else:
                return False
        EPS = NumericRange._EPS
        if abs(remainder(self.step, other.step)) > EPS:
            return False
        return abs(remainder(other.start - self.start, other.step)) <= EPS

    def normalize_bounds(self):
        """Normalizes this NumericRange.

        This returns a normalized range by reversing lb and ub if the
        NumericRange step is less than zero.  If lb and ub are
        reversed, then closed is updated to reflect that change.

        Returns
        -------
        lb, ub, closed

        """
        if self.step >= 0:
            return (self.start, self.end, self.closed)
        else:
            return (self.end, self.start, (self.closed[1], self.closed[0]))

    def _nooverlap(self, other):
        """Return True if the ranges for self and other are strictly separate"""
        s1, e1, c1 = self.normalize_bounds()
        s2, e2, c2 = other.normalize_bounds()
        if e1 < s2 or e2 < s1 or (e1 == s2 and (not (c1[1] and c2[0]))) or (e2 == s1 and (not (c2[1] and c1[0]))):
            return True
        return False

    @staticmethod
    def _split_ranges(cnr, new_step):
        """Split a discrete range into a list of ranges using a new step.

        This takes a single NumericRange and splits it into a set
        of new ranges, all of which use a new step.  The new_step must
        be a multiple of the current step.  CNR objects with a step of 0
        are returned unchanged.

        Parameters
        ----------
            cnr: `NumericRange`
                The range to split
            new_step: `int`
                The new step to use for returned ranges

        """
        if cnr.step == 0 or new_step == 0:
            return [cnr]
        assert new_step >= abs(cnr.step)
        assert new_step % cnr.step == 0
        _dir = int(math.copysign(1, cnr.step))
        _subranges = []
        for i in range(int(abs(new_step // cnr.step))):
            if _dir * (cnr.start + i * cnr.step) > _dir * cnr.end:
                break
            _subranges.append(NumericRange(cnr.start + i * cnr.step, cnr.end, _dir * new_step))
        return _subranges

    @staticmethod
    def _gcd(a, b):
        while b != 0:
            a, b = (b, a % b)
        return a

    @staticmethod
    def _lcm(a, b):
        gcd = NumericRange._gcd(a, b)
        if not gcd:
            return 0
        return a * b / gcd

    def _step_lcm(self, other_ranges):
        """This computes an approximate Least Common Multiple step"""
        if self.isdiscrete():
            a = self.step or 1
        else:
            a = 0
        for o in other_ranges:
            if o.isdiscrete():
                b = o.step or 1
            else:
                b = 0
            lcm = NumericRange._lcm(a, b)
            if lcm:
                a = lcm
            else:
                a += b
        return int(abs(a))

    def _push_to_discrete_element(self, val, push_to_next_larger_value):
        if not self.step or val in _infinite:
            return val
        else:
            if push_to_next_larger_value:
                _rndFcn = math.ceil if self.step > 0 else math.floor
            else:
                _rndFcn = math.floor if self.step > 0 else math.ceil
            return self.start + self.step * _rndFcn((val - self.start) / float(self.step))

    def range_difference(self, other_ranges):
        """Return the difference between this range and a list of other ranges.

        Parameters
        ----------
            other_ranges: `iterable`
                An iterable of other range objects to subtract from this range

        """
        _cnr_other_ranges = []
        for r in other_ranges:
            if isinstance(r, NumericRange):
                _cnr_other_ranges.append(r)
            elif type(r) is AnyRange:
                return []
            elif type(r) is NonNumericRange:
                continue
            else:
                raise ValueError('Unknown range type, %s' % (type(r).__name__,))
        other_ranges = _cnr_other_ranges
        lcm = self._step_lcm(other_ranges)
        _this = NumericRange._split_ranges(self, lcm)
        _other = []
        for s in other_ranges:
            _other.extend(NumericRange._split_ranges(s, lcm))
        for s in _other:
            _new_subranges = []
            for t in _this:
                if t._nooverlap(s):
                    _new_subranges.append(t)
                    continue
                if t.isdiscrete():
                    if s.isdiscrete() and (s.start - t.start) % lcm != 0:
                        _new_subranges.append(t)
                        continue
                t_min, t_max, t_c = t.normalize_bounds()
                s_min, s_max, s_c = s.normalize_bounds()
                if s.isdiscrete() and (not t.isdiscrete()):
                    if s_min == -_inf and t.start == -_inf or (s_max == _inf and t.end == _inf):
                        raise RangeDifferenceError('We do not support subtracting an infinite discrete range %s from an infinite continuous range %s' % (s, t))
                    start = max(s_min, s._push_to_discrete_element(t.start, True))
                    end = min(s_max, s._push_to_discrete_element(t.end, False))
                    if t.start < start:
                        _new_subranges.append(NumericRange(t.start, start, 0, (t.closed[0], False)))
                    if s.step:
                        for i in range(int((end - start) // s.step)):
                            _new_subranges.append(NumericRange(start + i * s.step, start + (i + 1) * s.step, 0, '()'))
                    if t.end > end:
                        _new_subranges.append(NumericRange(end, t.end, 0, (False, t.closed[1])))
                else:
                    if t_min < s_min:
                        if t.step:
                            s_min -= lcm
                            closed1 = True
                        _min = min(t_max, s_min)
                        if not t.step:
                            closed1 = not s_c[0] if _min is s_min else t_c[1]
                        _closed = (t_c[0], closed1)
                        _step = abs(t.step)
                        _rng = (t_min, _min)
                        if t_min == -_inf and t.step:
                            _step = -_step
                            _rng = (_rng[1], _rng[0])
                            _closed = (_closed[1], _closed[0])
                        _new_subranges.append(NumericRange(_rng[0], _rng[1], _step, _closed))
                    elif t_min == s_min and t_c[0] and (not s_c[0]):
                        _new_subranges.append(NumericRange(t_min, t_min, 0))
                    if t_max > s_max:
                        if t.step:
                            s_max += lcm
                            closed0 = True
                        _max = max(t_min, s_max)
                        if not t.step:
                            closed0 = not s_c[1] if _max is s_max else t_c[0]
                        _new_subranges.append(NumericRange(_max, t_max, abs(t.step), (closed0, t_c[1])))
                    elif t_max == s_max and t_c[1] and (not s_c[1]):
                        _new_subranges.append(NumericRange(t_max, t_max, 0))
            _this = _new_subranges
        return _this

    def range_intersection(self, other_ranges):
        """Return the intersection between this range and a set of other ranges.

        Parameters
        ----------
            other_ranges: `iterable`
                An iterable of other range objects to intersect with this range

        """
        _cnr_other_ranges = []
        for r in other_ranges:
            if isinstance(r, NumericRange):
                _cnr_other_ranges.append(r)
            elif type(r) is AnyRange:
                return [self]
            elif type(r) is NonNumericRange:
                continue
            else:
                raise ValueError('Unknown range type, %s' % (type(r).__name__,))
        other_ranges = _cnr_other_ranges
        lcm = self._step_lcm(other_ranges)
        ans = []
        _this = NumericRange._split_ranges(self, lcm)
        _other = []
        for s in other_ranges:
            _other.extend(NumericRange._split_ranges(s, lcm))
        for t in _this:
            for s in _other:
                if s.isdiscrete() and t.isdiscrete():
                    if (s.start - t.start) % lcm != 0:
                        continue
                if t._nooverlap(s):
                    continue
                t_min, t_max, t_c = t.normalize_bounds()
                s_min, s_max, s_c = s.normalize_bounds()
                step = abs(t.step if t.step else s.step)
                intersect_start = max(t._push_to_discrete_element(s_min, True), s._push_to_discrete_element(t_min, True))
                intersect_end = min(t._push_to_discrete_element(s_max, False), s._push_to_discrete_element(t_max, False))
                c = [True, True]
                if intersect_start == t_min:
                    c[0] &= t_c[0]
                if intersect_start == s_min:
                    c[0] &= s_c[0]
                if intersect_end == t_max:
                    c[1] &= t_c[1]
                if intersect_end == s_max:
                    c[1] &= s_c[1]
                if step and intersect_start == -_inf:
                    ans.append(NumericRange(intersect_end, intersect_start, -step, (c[1], c[0])))
                else:
                    ans.append(NumericRange(intersect_start, intersect_end, step, c))
        return ans