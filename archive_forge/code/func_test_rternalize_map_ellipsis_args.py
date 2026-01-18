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
@pytest.mark.parametrize('args', ((), (1,), (1, 2)))
def test_rternalize_map_ellipsis_args(args):

    def f(x, *args):
        return len(args)
    rfun = rinterface.rternalize(f, signature=True)
    assert ('x', '...') == tuple(rinterface.baseenv['formals'](rfun).names)
    assert rfun(0, *args)[0] == len(args)