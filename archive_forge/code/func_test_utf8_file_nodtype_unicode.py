import sys
import gc
import gzip
import os
import threading
import time
import warnings
import io
import re
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
from datetime import datetime
import locale
from multiprocessing import Value, get_context
from ctypes import c_bool
import numpy as np
import numpy.ma as ma
from numpy.lib._iotools import ConverterError, ConversionWarning
from numpy.compat import asbytes
from numpy.ma.testutils import assert_equal
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
def test_utf8_file_nodtype_unicode(self):
    utf8 = 'ϖ'
    latin1 = 'öüö'
    try:
        encoding = locale.getpreferredencoding()
        utf8.encode(encoding)
    except (UnicodeError, ImportError):
        pytest.skip('Skipping test_utf8_file_nodtype_unicode, unable to encode utf8 in preferred encoding')
    with temppath() as path:
        with io.open(path, 'wt') as f:
            f.write('norm1,norm2,norm3\n')
            f.write('norm1,' + latin1 + ',norm3\n')
            f.write('test1,testNonethe' + utf8 + ',test3\n')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(path, dtype=None, comments=None, delimiter=',')
            assert_(w[0].category is np.VisibleDeprecationWarning)
        ctl = np.array([['norm1', 'norm2', 'norm3'], ['norm1', latin1, 'norm3'], ['test1', 'testNonethe' + utf8, 'test3']], dtype=np.str_)
        assert_array_equal(test, ctl)