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
def test_parse_error_when_evaluting():
    with pytest.raises(_rinterface.RParsingError):
        rinterface.parse("list(''=1)")