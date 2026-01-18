from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
Confirm the intended behavior for *dtype* kwarg.

        The result of ``asarray()`` should have the dtype provided through the
        keyword argument, when used. This forces unique array handles to be
        produced for unique np.dtype objects, but (for equivalent dtypes), the
        underlying data (the base object) is shared with the original array
        object.

        Ref https://github.com/numpy/numpy/issues/1468
        