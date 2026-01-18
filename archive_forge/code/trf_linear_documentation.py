import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult
from .givens_elimination import givens_elimination
from .common import (
Select the best step according to Trust Region Reflective algorithm.