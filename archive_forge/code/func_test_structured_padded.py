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
def test_structured_padded(self):
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=[('foo', 'i4'), ('bar', 'i4'), ('baz', 'i4')])
    c = BytesIO()
    np.savetxt(c, a[['foo', 'baz']], fmt='%d')
    c.seek(0)
    assert_equal(c.readlines(), [b'1 3\n', b'4 6\n'])