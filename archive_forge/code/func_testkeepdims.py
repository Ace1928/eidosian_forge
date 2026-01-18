import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def testkeepdims(f, a, d):
    numpy_f = numpy.__getattribute__(f)
    ma_f = np.ma.__getattribute__(f)
    assert_equal(ma_f(a, keepdims=True).shape, numpy_f(d, keepdims=True).shape)
    assert_equal(ma_f(a, keepdims=False).shape, numpy_f(d, keepdims=False).shape)
    assert_equal(ma_f(a, axis=1, keepdims=True)[..., :-1], numpy_f(d[..., :-1], axis=1, keepdims=True))
    assert_equal(ma_f(a, axis=(0, 1), keepdims=True)[..., :-1], numpy_f(d[..., :-1], axis=(0, 1), keepdims=True))