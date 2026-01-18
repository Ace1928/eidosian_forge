import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult
from .common import (
Select the best step according to Trust Region Reflective algorithm.