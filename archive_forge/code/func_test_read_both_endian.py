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
def test_read_both_endian():
    for fname in ('big_endian.mat', 'little_endian.mat'):
        fp = open(pjoin(test_data_path, fname), 'rb')
        rdr = MatFile5Reader(fp)
        d = rdr.get_variables()
        fp.close()
        assert_array_equal(d['strings'], np.array([['hello'], ['world']], dtype=object))
        assert_array_equal(d['floats'], np.array([[2.0, 3.0], [3.0, 4.0]], dtype=np.float32))