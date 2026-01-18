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
def test_utf8_byte_encoding(self):
    utf8 = b'\xcf\x96'
    norm = b'norm1,norm2,norm3\n'
    enc = b'test1,testNonethe' + utf8 + b',test3\n'
    s = norm + enc + norm
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        test = np.genfromtxt(TextIO(s), dtype=None, comments=None, delimiter=',')
        assert_(w[0].category is np.VisibleDeprecationWarning)
    ctl = np.array([[b'norm1', b'norm2', b'norm3'], [b'test1', b'testNonethe' + utf8, b'test3'], [b'norm1', b'norm2', b'norm3']])
    assert_array_equal(test, ctl)