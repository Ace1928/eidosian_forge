from ...sage_helper import _within_sage
from ..upper_halfspace.finite_point import *
def _translate_to_match(self, z, targetZ):
    v = self._to_vec(targetZ - z)
    integers = [interval.is_int()[1] for interval in v]
    if None in integers:
        return None
    return z + integers[0] * self.t0 + integers[1] * self.t1