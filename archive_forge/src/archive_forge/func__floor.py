from ..sage_helper import _within_sage
import math
def _floor(x):
    if isinstance(x, float):
        return math.floor(x)
    return int(x.floor())