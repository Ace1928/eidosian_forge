import math
import textwrap
import sys
import pytest
import threading
import traceback
import time
import numpy as np
from numpy.testing import IS_PYPY
from . import util
@pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MinGW64 Gfortran (Issue #9673)')
def test_string_callback(self):

    def callback(code):
        if code == 'r':
            return 0
        else:
            return 1
    f = getattr(self.module, 'string_callback')
    r = f(callback)
    assert r == 0