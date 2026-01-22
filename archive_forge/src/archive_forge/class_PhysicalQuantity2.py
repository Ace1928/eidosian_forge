import pytest
from numpy import (
from numpy.testing import (
class PhysicalQuantity2(ndarray):
    __array_priority__ = 10