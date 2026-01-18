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
def test_recfromcsv(self):
    with temppath(suffix='.txt') as path:
        path = Path(path)
        with path.open('w') as f:
            f.write('A,B\n0,1\n2,3')
        kwargs = dict(missing_values='N/A', names=True, case_sensitive=True)
        test = np.recfromcsv(path, dtype=None, **kwargs)
        control = np.array([(0, 1), (2, 3)], dtype=[('A', int), ('B', int)])
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)