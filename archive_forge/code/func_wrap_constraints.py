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
def wrap_constraints(g):
    cons = []
    if g is not None:
        if not isinstance(g, (tuple, list)):
            g = (g,)
        else:
            pass
        for g in g:
            cons.append({'type': 'ineq', 'fun': g})
        cons = tuple(cons)
    else:
        cons = None
    return cons