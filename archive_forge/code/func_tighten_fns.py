import numpy as np
from cvxpy.atoms import ceil, floor, length
def tighten_fns(expr):
    if type(expr) in integer_valued_fns:
        return (np.ceil, np.floor)
    elif expr.is_nonneg():
        return (lambda t: np.maximum(t, 0), lambda t: t)
    elif expr.is_nonpos():
        return (lambda t: t, lambda t: np.minimum(t, 0))
    else:
        return (lambda t: t, lambda t: t)