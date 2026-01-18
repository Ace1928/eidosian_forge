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
def test_utf8_userconverters_with_explicit_dtype(self):
    utf8 = b'\xcf\x96'
    with temppath() as path:
        with open(path, 'wb') as f:
            f.write(b'skip,skip,2001-01-01' + utf8 + b',1.0,skip')
        test = np.genfromtxt(path, delimiter=',', names=None, dtype=float, usecols=(2, 3), converters={2: np.compat.unicode}, encoding='UTF-8')
    control = np.array([('2001-01-01' + utf8.decode('UTF-8'), 1.0)], dtype=[('', '|U11'), ('', float)])
    assert_equal(test, control)