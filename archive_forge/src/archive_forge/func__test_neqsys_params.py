from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _test_neqsys_params(solver, **kwargs):
    ns = NeqSys(2, 2, f, jac=j)
    x, sol = ns.solve([0, 0], [3], solver=solver, **kwargs)
    assert abs(x[0] - 0.8411639) < 2e-07
    assert abs(x[1] - 0.1588361) < 2e-07