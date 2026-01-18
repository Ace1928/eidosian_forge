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
def test_single_dtype_wo_names(self):
    data = '0, 1, 2.3\n4, 5, 6.7'
    mtest = np.genfromtxt(TextIO(data), delimiter=',', dtype=float, defaultfmt='f%02i')
    ctrl = np.array([[0.0, 1.0, 2.3], [4.0, 5.0, 6.7]], dtype=float)
    assert_equal(mtest, ctrl)