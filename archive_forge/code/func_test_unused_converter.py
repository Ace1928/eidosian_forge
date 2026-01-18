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
def test_unused_converter(self):
    data = TextIO('1 21\n  3 42\n')
    test = np.genfromtxt(data, usecols=(1,), converters={0: lambda s: int(s, 16)})
    assert_equal(test, [21, 42])
    data.seek(0)
    test = np.genfromtxt(data, usecols=(1,), converters={1: lambda s: int(s, 16)})
    assert_equal(test, [33, 66])