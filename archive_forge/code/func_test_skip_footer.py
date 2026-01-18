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
def test_skip_footer(self):
    data = ['# %i' % i for i in range(1, 6)]
    data.append('A, B, C')
    data.extend(['%i,%3.1f,%03s' % (i, i, i) for i in range(51)])
    data[-1] = '99,99'
    kwargs = dict(delimiter=',', names=True, skip_header=5, skip_footer=10)
    test = np.genfromtxt(TextIO('\n'.join(data)), **kwargs)
    ctrl = np.array([('%f' % i, '%f' % i, '%f' % i) for i in range(41)], dtype=[(_, float) for _ in 'ABC'])
    assert_equal(test, ctrl)