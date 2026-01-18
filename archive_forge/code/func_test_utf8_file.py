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
def test_utf8_file(self):
    utf8 = b'\xcf\x96'
    with temppath() as path:
        with open(path, 'wb') as f:
            f.write((b'test1,testNonethe' + utf8 + b',test3\n') * 2)
        test = np.genfromtxt(path, dtype=None, comments=None, delimiter=',', encoding='UTF-8')
        ctl = np.array([['test1', 'testNonethe' + utf8.decode('UTF-8'), 'test3'], ['test1', 'testNonethe' + utf8.decode('UTF-8'), 'test3']], dtype=np.str_)
        assert_array_equal(test, ctl)
        with open(path, 'wb') as f:
            f.write(b'0,testNonethe' + utf8)
        test = np.genfromtxt(path, dtype=None, comments=None, delimiter=',', encoding='UTF-8')
        assert_equal(test['f0'], 0)
        assert_equal(test['f1'], 'testNonethe' + utf8.decode('UTF-8'))