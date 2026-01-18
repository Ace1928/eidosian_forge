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
@pytest.mark.skipif(IS_WASM, reason='Cannot start thread')
def test_savez_filename_clashes(self):

    def writer(error_list):
        with temppath(suffix='.npz') as tmp:
            arr = np.random.randn(500, 500)
            try:
                np.savez(tmp, arr=arr)
            except OSError as err:
                error_list.append(err)
    errors = []
    threads = [threading.Thread(target=writer, args=(errors,)) for j in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise AssertionError(errors)