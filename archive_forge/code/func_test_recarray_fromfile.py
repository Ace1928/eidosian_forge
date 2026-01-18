import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_fromfile(self):
    data_dir = path.join(path.dirname(__file__), 'data')
    filename = path.join(data_dir, 'recarray_from_file.fits')
    fd = open(filename, 'rb')
    fd.seek(2880 * 2)
    r1 = np.rec.fromfile(fd, formats='f8,i4,a5', shape=3, byteorder='big')
    fd.seek(2880 * 2)
    r2 = np.rec.array(fd, formats='f8,i4,a5', shape=3, byteorder='big')
    fd.seek(2880 * 2)
    bytes_array = BytesIO()
    bytes_array.write(fd.read())
    bytes_array.seek(0)
    r3 = np.rec.fromfile(bytes_array, formats='f8,i4,a5', shape=3, byteorder='big')
    fd.close()
    assert_equal(r1, r2)
    assert_equal(r2, r3)