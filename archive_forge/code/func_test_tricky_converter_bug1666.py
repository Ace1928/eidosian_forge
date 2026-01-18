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
def test_tricky_converter_bug1666(self):
    s = TextIO('q1,2\nq3,4')
    cnv = lambda s: float(s[1:])
    test = np.genfromtxt(s, delimiter=',', converters={0: cnv})
    control = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_equal(test, control)