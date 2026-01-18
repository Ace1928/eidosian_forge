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
def test_rpy_memory():
    x = rinterface.IntSexpVector(range(10))
    x_rid = x.rid
    assert x_rid in set((z[0] for z in rinterface._rinterface.protected_rids()))
    del x
    gc.collect()
    gc.collect()
    s = set((z[0] for z in rinterface._rinterface.protected_rids()))
    assert x_rid not in s