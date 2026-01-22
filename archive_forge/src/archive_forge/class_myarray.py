import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class myarray(np.ndarray):
    __array_priority__ = 1.0