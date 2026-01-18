import numpy as np
def to_unrestricted(p, sel, bounds):
    """
    Transform parameters to the unrestricted [0,1] space

    Parameters
    ----------
    p : ndarray
        Parameters that strictly satisfy the constraints

    Returns
    -------
    ndarray
        Parameters all in (0,1)
    """
    a, b, g = p[:3]
    if sel[0]:
        lb = max(LOWER_BOUND, bounds[0, 0])
        ub = min(1 - LOWER_BOUND, bounds[0, 1])
        a = (a - lb) / (ub - lb)
    if sel[1]:
        lb = bounds[1, 0]
        ub = min(p[0], bounds[1, 1])
        b = (b - lb) / (ub - lb)
    if sel[2]:
        lb = bounds[2, 0]
        ub = min(1.0 - p[0], bounds[2, 1])
        g = (g - lb) / (ub - lb)
    return (a, b, g)