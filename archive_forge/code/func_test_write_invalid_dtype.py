import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.io import netcdf_file
from scipy._lib._tmpdirs import in_tempdir
def test_write_invalid_dtype():
    dtypes = ['int64', 'uint64']
    if np.dtype('int').itemsize == 8:
        dtypes.append('int')
    if np.dtype('uint').itemsize == 8:
        dtypes.append('uint')
    with netcdf_file(BytesIO(), 'w') as f:
        f.createDimension('time', N_EG_ELS)
        for dt in dtypes:
            assert_raises(ValueError, f.createVariable, 'time', dt, ('time',))