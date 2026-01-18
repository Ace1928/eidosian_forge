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
@pytest.mark.parametrize('fmt', ['%f', b'%f'])
@pytest.mark.parametrize('iotype', [StringIO, BytesIO])
def test_unicode_and_bytes_fmt(self, fmt, iotype):
    a = np.array([1.0])
    s = iotype()
    np.savetxt(s, a, fmt=fmt)
    s.seek(0)
    if iotype is StringIO:
        assert_equal(s.read(), '%f\n' % 1.0)
    else:
        assert_equal(s.read(), b'%f\n' % 1.0)