import gc
import multiprocessing
import os
import pickle
import pytest
from rpy2 import rinterface
import rpy2
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import signal
import sys
import subprocess
import tempfile
import textwrap
import time
def test_rternalize_namedargs():

    def f(x, y, z=None):
        if z is None:
            return x[0] + y[0]
        else:
            return z[0]
    rfun = rinterface.rternalize(f, signature=False)
    res = rfun(1, 2)
    assert res[0] == 3
    res = rfun(1, 2, z=8)
    assert res[0] == 8