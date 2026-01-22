from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def F_inner(self, x):
    return (x[0] - 1) ** 2 + (x[1] - 1) ** 2