import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_pickle_python2_python3():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    expected = np.array([None, range, '優良', b'\xe4\xb8\x8d\xe8\x89\xaf'], dtype=object)
    for fname in ['py2-objarr.npy', 'py2-objarr.npz', 'py3-objarr.npy', 'py3-objarr.npz']:
        path = os.path.join(data_dir, fname)
        for encoding in ['bytes', 'latin1']:
            data_f = np.load(path, allow_pickle=True, encoding=encoding)
            if fname.endswith('.npz'):
                data = data_f['x']
                data_f.close()
            else:
                data = data_f
            if encoding == 'latin1' and fname.startswith('py2'):
                assert_(isinstance(data[3], str))
                assert_array_equal(data[:-1], expected[:-1])
                assert_array_equal(data[-1].encode(encoding), expected[-1])
            else:
                assert_(isinstance(data[3], bytes))
                assert_array_equal(data, expected)
        if fname.startswith('py2'):
            if fname.endswith('.npz'):
                data = np.load(path, allow_pickle=True)
                assert_raises(UnicodeError, data.__getitem__, 'x')
                data.close()
                data = np.load(path, allow_pickle=True, fix_imports=False, encoding='latin1')
                assert_raises(ImportError, data.__getitem__, 'x')
                data.close()
            else:
                assert_raises(UnicodeError, np.load, path, allow_pickle=True)
                assert_raises(ImportError, np.load, path, allow_pickle=True, fix_imports=False, encoding='latin1')