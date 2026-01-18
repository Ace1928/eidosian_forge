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
def test_str_round():
    stream = BytesIO()
    in_arr = np.array(['Hello', 'Foob'])
    out_arr = np.array(['Hello', 'Foob '])
    savemat(stream, dict(a=in_arr))
    res = loadmat(stream)
    assert_array_equal(res['a'], out_arr)
    stream.truncate(0)
    stream.seek(0)
    in_str = in_arr.tobytes(order='F')
    in_from_str = np.ndarray(shape=a.shape, dtype=in_arr.dtype, order='F', buffer=in_str)
    savemat(stream, dict(a=in_from_str))
    assert_array_equal(res['a'], out_arr)
    stream.truncate(0)
    stream.seek(0)
    in_arr_u = in_arr.astype('U')
    out_arr_u = out_arr.astype('U')
    savemat(stream, {'a': in_arr_u})
    res = loadmat(stream)
    assert_array_equal(res['a'], out_arr_u)