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
def test_not_closing_opened_fid(self):
    with temppath(suffix='.npz') as tmp:
        with open(tmp, 'wb') as fp:
            np.savez(fp, data='LOVELY LOAD')
        with open(tmp, 'rb', 10000) as fp:
            fp.seek(0)
            assert_(not fp.closed)
            np.load(fp)['data']
            assert_(not fp.closed)
            fp.seek(0)
            assert_(not fp.closed)