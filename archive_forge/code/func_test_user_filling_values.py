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
def test_user_filling_values(self):
    ctrl = np.array([(0, 3), (4, -999)], dtype=[('a', int), ('b', int)])
    data = 'N/A, 2, 3\n4, ,???'
    kwargs = dict(delimiter=',', dtype=int, names='a,b,c', missing_values={0: 'N/A', 'b': ' ', 2: '???'}, filling_values={0: 0, 'b': 0, 2: -999})
    test = np.genfromtxt(TextIO(data), **kwargs)
    ctrl = np.array([(0, 2, 3), (4, 0, -999)], dtype=[(_, int) for _ in 'abc'])
    assert_equal(test, ctrl)
    test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
    ctrl = np.array([(0, 3), (4, -999)], dtype=[(_, int) for _ in 'ac'])
    assert_equal(test, ctrl)
    data2 = '1,2,*,4\n5,*,7,8\n'
    test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int, missing_values='*', filling_values=0)
    ctrl = np.array([[1, 2, 0, 4], [5, 0, 7, 8]])
    assert_equal(test, ctrl)
    test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int, missing_values='*', filling_values=-1)
    ctrl = np.array([[1, 2, -1, 4], [5, -1, 7, 8]])
    assert_equal(test, ctrl)