from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def _interval_minimum_candidates(intervals_and_extras):
    result = [intervals_and_extras[0]]
    for this in intervals_and_extras[1:]:
        t0 = this[0]
        if not all((t0 > other[0] for other in result)):
            if all((t0 < other[0] for other in result)):
                result = [this]
            else:
                result.append(this)
    return result