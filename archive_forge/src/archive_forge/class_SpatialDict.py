from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence
class SpatialDict:
    """
    A python dict-like object appropriate for using numerical points (e.g.,
    in the hyperboloid model) as keys. That is, look-ups return
    the same entry for points that are almost but not exactly the
    same due to rounding-errors.

    To achieve this, the points are assumed to be in some lattice
    and the minimal distance between any two points in the lattice
    must be given.
    """
    _scale = 1024

    def __init__(self, min_distance, verified):
        RF = min_distance.parent()
        self._min_distance = min_distance
        self._RF_scale = RF(self._scale)
        if verified:
            self._right_distance_value = min_distance
            self._left_distance_value = 0
        else:
            self._right_distance_value = min_distance * RF(0.125)
            self._left_distance_value = min_distance * RF(0.5)
        self._data = {}

    def setdefault(self, point, default):
        reps_and_ikeys = self._representatives_and_ikeys(point)
        for rep, ikey in reps_and_ikeys:
            for other_rep, entry in self._data.get(ikey, []):
                d = self.distance(rep, other_rep)
                if d < self._right_distance_value:
                    return entry.value
                if not self._left_distance_value < d:
                    raise InsufficientPrecisionError('Could neither verify that the two given tiles are the same nor that they are distinct. Distance between basepoint translates is: %r. Injectivty diameter about basepoint is: %r.' % (d, self._min_distance))
        entry = _Entry(default)
        for rep, ikey in reps_and_ikeys:
            self._data.setdefault(ikey, []).append((rep, entry))
        return default

    def distance(self, point_0, point_1):
        raise NotImplementedError()

    def representatives(self, point):
        return [point]

    def float_hash(self, point):
        raise NotImplementedError()

    def _representatives_and_ikeys(self, point):
        return [(rep, ikey) for rep in self.representatives(point) for ikey in floor_as_integers(self._RF_scale * self.float_hash(rep))]