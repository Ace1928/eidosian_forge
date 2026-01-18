import os
import sys
from tempfile import mkdtemp
from shutil import rmtree
import pytest
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
@pytest.mark.skipif(sys.version_info >= (3, 8), reason='multiprocessing issues in Python 3.8')
def test_run_multiproc_nondaemon_true():
    result = run_multiproc_nondaemon_with_flag(True)
    assert result == 180