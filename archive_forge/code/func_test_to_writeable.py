import os
from collections import OrderedDict
from os.path import join as pjoin, dirname
from glob import glob
from io import BytesIO
import re
from tempfile import mkdtemp
import warnings
import shutil
import gzip
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import array
import scipy.sparse as SP
import scipy.io
from scipy.io.matlab import MatlabOpaque, MatlabFunction, MatlabObject
import scipy.io.matlab._byteordercodes as boc
from scipy.io.matlab._miobase import (
from scipy.io.matlab._mio import mat_reader_factory, loadmat, savemat, whosmat
from scipy.io.matlab._mio5 import (
import scipy.io.matlab._mio5_params as mio5p
from scipy._lib._util import VisibleDeprecationWarning
def test_to_writeable():
    res = to_writeable(np.array([1]))
    assert_equal(res.shape, (1,))
    assert_array_equal(res, 1)
    expected1 = np.array([(1, 2)], dtype=[('a', '|O8'), ('b', '|O8')])
    expected2 = np.array([(2, 1)], dtype=[('b', '|O8'), ('a', '|O8')])
    alternatives = (expected1, expected2)
    assert_any_equal(to_writeable({'a': 1, 'b': 2}), alternatives)
    assert_any_equal(to_writeable({'a': 1, 'b': 2, '_c': 3}), alternatives)
    assert_any_equal(to_writeable({'a': 1, 'b': 2, 100: 3}), alternatives)
    assert_any_equal(to_writeable({'a': 1, 'b': 2, '99': 3}), alternatives)

    class klass:
        pass
    c = klass
    c.a = 1
    c.b = 2
    assert_any_equal(to_writeable(c), alternatives)
    res = to_writeable([])
    assert_equal(res.shape, (0,))
    assert_equal(res.dtype.type, np.float64)
    res = to_writeable(())
    assert_equal(res.shape, (0,))
    assert_equal(res.dtype.type, np.float64)
    assert_(to_writeable(None) is None)
    assert_equal(to_writeable('a string').dtype.type, np.str_)
    res = to_writeable(1)
    assert_equal(res.shape, ())
    assert_equal(res.dtype.type, np.array(1).dtype.type)
    assert_array_equal(res, 1)
    assert_(to_writeable({}) is EmptyStructMarker)
    assert_(to_writeable(object()) is None)

    class C:
        pass
    assert_(to_writeable(c()) is EmptyStructMarker)
    res = to_writeable({'a': 1})['a']
    assert_equal(res.shape, (1,))
    assert_equal(res.dtype.type, np.object_)
    assert_(to_writeable({'1': 1}) is EmptyStructMarker)
    assert_(to_writeable({'_a': 1}) is EmptyStructMarker)
    assert_equal(to_writeable({'1': 1, 'f': 2}), np.array([(2,)], dtype=[('f', '|O8')]))