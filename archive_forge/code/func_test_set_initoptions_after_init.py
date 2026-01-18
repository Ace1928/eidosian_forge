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
def test_set_initoptions_after_init():
    with pytest.raises(RuntimeError):
        rinterface.embedded.set_initoptions(('aa', '--verbose', '--no-save'))