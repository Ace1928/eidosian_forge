from collections import namedtuple
import math
import warnings
@property
def _scaling(self):
    """The absolute scaling factors of the transformation.

        This tuple represents the absolute value of the scaling factors of the
        transformation, sorted from bigger to smaller.
        """
    a, b, _, d, e, _, _, _, _ = self
    trace = a ** 2 + b ** 2 + d ** 2 + e ** 2
    det = (a * e - b * d) ** 2
    delta = trace ** 2 / 4 - det
    if delta < 1e-12:
        delta = 0
    l1 = math.sqrt(trace / 2 + math.sqrt(delta))
    l2 = math.sqrt(trace / 2 - math.sqrt(delta))
    return (l1, l2)