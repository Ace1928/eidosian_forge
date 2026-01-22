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
def _check_level(label, expected, actual):
    """ Check one level of a potentially nested array """
    if SP.issparse(expected):
        assert_(SP.issparse(actual))
        assert_array_almost_equal(actual.toarray(), expected.toarray(), err_msg=label, decimal=5)
        return
    assert_(types_compatible(expected, actual), f'Expected type {type(expected)}, got {type(actual)} at {label}')
    if not isinstance(expected, (np.void, np.ndarray, MatlabObject)):
        assert_equal(expected, actual)
        return
    assert_(expected.shape == actual.shape, msg=f'Expected shape {expected.shape}, got {actual.shape} at {label}')
    ex_dtype = expected.dtype
    if ex_dtype.hasobject:
        if isinstance(expected, MatlabObject):
            assert_equal(expected.classname, actual.classname)
        for i, ev in enumerate(expected):
            level_label = '%s, [%d], ' % (label, i)
            _check_level(level_label, ev, actual[i])
        return
    if ex_dtype.fields:
        for fn in ex_dtype.fields:
            level_label = f'{label}, field {fn}, '
            _check_level(level_label, expected[fn], actual[fn])
        return
    if ex_dtype.type in (str, np.str_, np.bool_):
        assert_equal(actual, expected, err_msg=label)
        return
    assert_array_almost_equal(actual, expected, err_msg=label, decimal=5)