import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def x_only(x):
    cosx = np.cos(x)
    sinx = np.sin(x)
    return np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])