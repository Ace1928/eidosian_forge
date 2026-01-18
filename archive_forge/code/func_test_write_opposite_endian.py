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
def test_write_opposite_endian():
    float_arr = np.array([[2.0, 3.0], [3.0, 4.0]])
    int_arr = np.arange(6).reshape((2, 3))
    uni_arr = np.array(['hello', 'world'], dtype='U')
    stream = BytesIO()
    savemat(stream, {'floats': float_arr.byteswap().view(float_arr.dtype.newbyteorder()), 'ints': int_arr.byteswap().view(int_arr.dtype.newbyteorder()), 'uni_arr': uni_arr.byteswap().view(uni_arr.dtype.newbyteorder())})
    rdr = MatFile5Reader(stream)
    d = rdr.get_variables()
    assert_array_equal(d['floats'], float_arr)
    assert_array_equal(d['ints'], int_arr)
    assert_array_equal(d['uni_arr'], uni_arr)
    stream.close()