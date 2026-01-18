import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sequence', [[np.array(1), np.array(2)], [[1, 2], [3, 4]]])
def test_to_object_array(sequence):
    out = _to_object_array(sequence)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == 'O'
    assert out.ndim == 1