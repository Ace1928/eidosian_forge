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
@pytest.mark.parametrize('kwargs', ({}, {'y': 1}, {'y': 1, 'z': 2}))
def test_rternalize_map_ellipsis_kwargs(kwargs):

    def f(x, **kwargs):
        return len(kwargs)
    rfun = rinterface.rternalize(f, signature=True)
    assert ('x', '...') == tuple(rinterface.baseenv['formals'](rfun).names)
    assert rfun(0, **kwargs)[0] == len(kwargs)