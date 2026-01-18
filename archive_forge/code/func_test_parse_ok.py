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
def test_parse_ok():
    xp = rinterface.parse('2 + 3')
    assert xp.typeof == rinterface.RTYPES.EXPRSXP
    assert 2.0 == xp[0][1][0]
    assert 3.0 == xp[0][2][0]