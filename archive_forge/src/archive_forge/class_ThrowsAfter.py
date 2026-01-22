import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class ThrowsAfter:

    def __init__(self, iters):
        self.iters_left = iters

    def __bool__(self):
        if self.iters_left == 0:
            raise ValueError('called `iters` times')
        self.iters_left -= 1
        return True