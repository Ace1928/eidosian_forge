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
def test_user_missing_values(self):
    data = 'A, B, C\n0, 0., 0j\n1, N/A, 1j\n-9, 2.2, N/A\n3, -99, 3j'
    basekwargs = dict(dtype=None, delimiter=',', names=True)
    mdtype = [('A', int), ('B', float), ('C', complex)]
    test = np.genfromtxt(TextIO(data), missing_values='N/A', **basekwargs)
    control = ma.array([(0, 0.0, 0j), (1, -999, 1j), (-9, 2.2, -999j), (3, -99, 3j)], mask=[(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)], dtype=mdtype)
    assert_equal(test, control)
    basekwargs['dtype'] = mdtype
    test = np.genfromtxt(TextIO(data), missing_values={0: -9, 1: -99, 2: -999j}, usemask=True, **basekwargs)
    control = ma.array([(0, 0.0, 0j), (1, -999, 1j), (-9, 2.2, -999j), (3, -99, 3j)], mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)], dtype=mdtype)
    assert_equal(test, control)
    test = np.genfromtxt(TextIO(data), missing_values={0: -9, 'B': -99, 'C': -999j}, usemask=True, **basekwargs)
    control = ma.array([(0, 0.0, 0j), (1, -999, 1j), (-9, 2.2, -999j), (3, -99, 3j)], mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)], dtype=mdtype)
    assert_equal(test, control)