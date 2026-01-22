import numpy as np
def _cabs(x):
    """absolute value function that changes complex sign based on real sign

    This could be useful for complex step derivatives of functions that
    need abs. Not yet used.
    """
    sign = (x.real >= 0) * 2 - 1
    return sign * x