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
def test_withmissing(self):
    data = TextIO('A,B\n0,1\n2,N/A')
    kwargs = dict(delimiter=',', missing_values='N/A', names=True)
    test = np.genfromtxt(data, dtype=None, usemask=True, **kwargs)
    control = ma.array([(0, 1), (2, -1)], mask=[(False, False), (False, True)], dtype=[('A', int), ('B', int)])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)
    data.seek(0)
    test = np.genfromtxt(data, usemask=True, **kwargs)
    control = ma.array([(0, 1), (2, -1)], mask=[(False, False), (False, True)], dtype=[('A', float), ('B', float)])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)