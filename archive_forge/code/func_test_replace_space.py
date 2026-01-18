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
def test_replace_space(self):
    txt = 'A.A, B (B), C:C\n1, 2, 3.14'
    test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=None)
    ctrl_dtype = [('AA', int), ('B_B', int), ('CC', float)]
    ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
    assert_equal(test, ctrl)
    test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=None, replace_space='', deletechars='')
    ctrl_dtype = [('A.A', int), ('B (B)', int), ('C:C', float)]
    ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
    assert_equal(test, ctrl)
    test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=None, deletechars='')
    ctrl_dtype = [('A.A', int), ('B_(B)', int), ('C:C', float)]
    ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
    assert_equal(test, ctrl)