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
def test_save_object():

    class C:
        pass
    c = C()
    c.field1 = 1
    c.field2 = 'a string'
    stream = BytesIO()
    savemat(stream, {'c': c})
    d = loadmat(stream, struct_as_record=False)
    c2 = d['c'][0, 0]
    assert_equal(c2.field1, 1)
    assert_equal(c2.field2, 'a string')
    d = loadmat(stream, struct_as_record=True)
    c2 = d['c'][0, 0]
    assert_equal(c2['field1'], 1)
    assert_equal(c2['field2'], 'a string')