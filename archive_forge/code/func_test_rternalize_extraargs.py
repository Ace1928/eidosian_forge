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
@pytest.mark.parametrize('signature', ((True,), (False,)))
def test_rternalize_extraargs(signature):

    def f():
        return 1
    rfun = rinterface.rternalize(f, signature=signature)
    assert rfun()[0] == 1
    with pytest.raises(rinterface.embedded.RRuntimeError, match='unused argument \\(1\\)'):
        rfun(1)