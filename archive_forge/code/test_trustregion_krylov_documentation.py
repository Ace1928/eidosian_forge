import numpy as np
from scipy.optimize._trlib import (get_trlib_quadratic_subproblem)
from numpy.testing import (assert_,

Unit tests for Krylov space trust-region subproblem solver.

To run it in its simplest form::
  nosetests test_optimize.py

