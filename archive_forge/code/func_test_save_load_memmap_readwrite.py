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
@pytest.mark.xfail(IS_WASM, reason="memmap doesn't work correctly")
def test_save_load_memmap_readwrite(self):
    with temppath(suffix='.npy') as path:
        path = Path(path)
        a = np.array([[1, 2], [3, 4]], int)
        np.save(path, a)
        b = np.load(path, mmap_mode='r+')
        a[0][0] = 5
        b[0][0] = 5
        del b
        if IS_PYPY:
            break_cycles()
            break_cycles()
        data = np.load(path)
        assert_array_equal(data, a)