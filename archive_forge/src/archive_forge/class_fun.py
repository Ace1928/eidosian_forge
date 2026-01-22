from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np
from scipy.optimize import root
class fun:

    def __init__(self):
        self.count = 0

    def __call__(self, x):
        self.count += 1
        if not self.count % 5:
            ret = x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0
        else:
            ret = [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]
        return ret