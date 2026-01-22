import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
class StructTestS(StructTestFunction):

    def f(self, x):
        return (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.5) ** 2 + (x[3] - 0.5) ** 2
    g = None
    cons = wrap_constraints(g)