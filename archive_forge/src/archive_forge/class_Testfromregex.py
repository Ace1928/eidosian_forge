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
class Testfromregex:

    def test_record(self):
        c = TextIO()
        c.write('1.312 foo\n1.534 bar\n4.444 qux')
        c.seek(0)
        dt = [('num', np.float64), ('val', 'S3')]
        x = np.fromregex(c, '([0-9.]+)\\s+(...)', dt)
        a = np.array([(1.312, 'foo'), (1.534, 'bar'), (4.444, 'qux')], dtype=dt)
        assert_array_equal(x, a)

    def test_record_2(self):
        c = TextIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)
        dt = [('num', np.int32), ('val', 'S3')]
        x = np.fromregex(c, '(\\d+)\\s+(...)', dt)
        a = np.array([(1312, 'foo'), (1534, 'bar'), (4444, 'qux')], dtype=dt)
        assert_array_equal(x, a)

    def test_record_3(self):
        c = TextIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)
        dt = [('num', np.float64)]
        x = np.fromregex(c, '(\\d+)\\s+...', dt)
        a = np.array([(1312,), (1534,), (4444,)], dtype=dt)
        assert_array_equal(x, a)

    @pytest.mark.parametrize('path_type', [str, Path])
    def test_record_unicode(self, path_type):
        utf8 = b'\xcf\x96'
        with temppath() as str_path:
            path = path_type(str_path)
            with open(path, 'wb') as f:
                f.write(b'1.312 foo' + utf8 + b' \n1.534 bar\n4.444 qux')
            dt = [('num', np.float64), ('val', 'U4')]
            x = np.fromregex(path, '(?u)([0-9.]+)\\s+(\\w+)', dt, encoding='UTF-8')
            a = np.array([(1.312, 'foo' + utf8.decode('UTF-8')), (1.534, 'bar'), (4.444, 'qux')], dtype=dt)
            assert_array_equal(x, a)
            regexp = re.compile('([0-9.]+)\\s+(\\w+)', re.UNICODE)
            x = np.fromregex(path, regexp, dt, encoding='UTF-8')
            assert_array_equal(x, a)

    def test_compiled_bytes(self):
        regexp = re.compile(b'(\\d)')
        c = BytesIO(b'123')
        dt = [('num', np.float64)]
        a = np.array([1, 2, 3], dtype=dt)
        x = np.fromregex(c, regexp, dt)
        assert_array_equal(x, a)

    def test_bad_dtype_not_structured(self):
        regexp = re.compile(b'(\\d)')
        c = BytesIO(b'123')
        with pytest.raises(TypeError, match='structured datatype'):
            np.fromregex(c, regexp, dtype=np.float64)