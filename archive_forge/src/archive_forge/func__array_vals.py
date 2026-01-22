import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def _array_vals():
    for d in _integer_dtypes:
        yield asarray(1, dtype=d)
    for d in _boolean_dtypes:
        yield asarray(False, dtype=d)
    for d in _floating_dtypes:
        yield asarray(1.0, dtype=d)