import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from multiprocessing.dummy import Pool
def _lorenzian(x):
    return 1 / (1 + x ** 2)