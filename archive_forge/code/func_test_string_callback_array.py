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
def test_string_callback_array(self):
    cu1 = np.zeros((1,), 'S8')
    cu2 = np.zeros((1, 8), 'c')
    cu3 = np.array([''], 'S8')

    def callback(cu, lencu):
        if cu.shape != (lencu,):
            return 1
        if cu.dtype != 'S8':
            return 2
        if not np.all(cu == b''):
            return 3
        return 0
    f = getattr(self.module, 'string_callback_array')
    for cu in [cu1, cu2, cu3]:
        res = f(callback, cu, cu.size)
        assert res == 0