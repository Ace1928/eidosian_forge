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
def test_unserialize():
    x = rinterface.IntSexpVector([1, 2, 3])
    x_serialized = x.__getstate__()
    x_again = rinterface.Sexp(rinterface.unserialize(x_serialized))
    identical = rinterface.baseenv['identical']
    assert not x.rsame(x_again)
    assert identical(x, x_again)[0]