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
def test_skip_footer_with_invalid(self):
    with suppress_warnings() as sup:
        sup.filter(ConversionWarning)
        basestr = '1 1\n2 2\n3 3\n4 4\n5  \n6  \n7  \n'
        assert_raises(ValueError, np.genfromtxt, TextIO(basestr), skip_footer=1)
        a = np.genfromtxt(TextIO(basestr), skip_footer=1, invalid_raise=False)
        assert_equal(a, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))
        a = np.genfromtxt(TextIO(basestr), skip_footer=3)
        assert_equal(a, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))
        basestr = '1 1\n2  \n3 3\n4 4\n5  \n6 6\n7 7\n'
        a = np.genfromtxt(TextIO(basestr), skip_footer=1, invalid_raise=False)
        assert_equal(a, np.array([[1.0, 1.0], [3.0, 3.0], [4.0, 4.0], [6.0, 6.0]]))
        a = np.genfromtxt(TextIO(basestr), skip_footer=3, invalid_raise=False)
        assert_equal(a, np.array([[1.0, 1.0], [3.0, 3.0], [4.0, 4.0]]))