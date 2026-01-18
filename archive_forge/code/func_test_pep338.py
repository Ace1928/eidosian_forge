import sys
import os
import pytest
from os.path import join as pathjoin, isfile, dirname
import subprocess
import numpy as np
from numpy.testing import assert_equal, IS_WASM
@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
def test_pep338():
    stdout = subprocess.check_output([sys.executable, '-mnumpy.f2py', '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))