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
def test_withmissing_float(self):
    data = TextIO('A,B\n0,1.5\n2,-999.00')
    test = np.genfromtxt(data, dtype=None, delimiter=',', missing_values='-999.0', names=True, usemask=True)
    control = ma.array([(0, 1.5), (2, -1.0)], mask=[(False, False), (False, True)], dtype=[('A', int), ('B', float)])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)