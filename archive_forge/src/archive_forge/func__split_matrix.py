import numpy as np
import operator
from . import (linear_sum_assignment, OptimizeResult)
from ._optimize import _check_unknown_options
from scipy._lib._util import check_random_state
import itertools
def _split_matrix(X, n):
    upper, lower = (X[:n], X[n:])
    return (upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:])