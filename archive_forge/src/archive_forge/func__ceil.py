from ..sage_helper import _within_sage
import math
def _ceil(x):
    if isinstance(x, float):
        return math.ceil(x)
    return int(x.ceil())