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
def test_object_dispatch_lang():
    formula = rinterface.globalenv.find('formula')
    obj = formula(rinterface.StrSexpVector(['y ~ x']))
    assert isinstance(obj, rinterface.SexpVector)
    assert obj.typeof == rinterface.RTYPES.LANGSXP