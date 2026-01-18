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
def test_unpack_auto_dtype(self):
    txt = TextIO('M 21 72.\nF 35 58.')
    expected = (np.array(['M', 'F']), np.array([21, 35]), np.array([72.0, 58.0]))
    test = np.genfromtxt(txt, dtype=None, unpack=True, encoding='utf-8')
    for arr, result in zip(expected, test):
        assert_array_equal(arr, result)
        assert_equal(arr.dtype, result.dtype)