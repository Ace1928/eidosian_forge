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
def test_stringload(self):
    nonascii = b'\xc3\xb6\xc3\xbc\xc3\xb6'.decode('UTF-8')
    with temppath() as path:
        with open(path, 'wb') as f:
            f.write(nonascii.encode('UTF-16'))
        x = self.loadfunc(path, encoding='UTF-16', dtype=np.str_)
        assert_array_equal(x, nonascii)