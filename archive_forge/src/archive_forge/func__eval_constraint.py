import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, concatenate, finfo,
from ._optimize import (OptimizeResult, _check_unknown_options,
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import atleast_nd, array_namespace
from numpy import exp, inf  # noqa: F401
def _eval_constraint(x, cons):
    if cons['eq']:
        c_eq = concatenate([atleast_1d(con['fun'](x, *con['args'])) for con in cons['eq']])
    else:
        c_eq = zeros(0)
    if cons['ineq']:
        c_ieq = concatenate([atleast_1d(con['fun'](x, *con['args'])) for con in cons['ineq']])
    else:
        c_ieq = zeros(0)
    c = concatenate((c_eq, c_ieq))
    return c