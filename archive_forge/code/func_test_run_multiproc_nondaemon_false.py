import os
import sys
from tempfile import mkdtemp
from shutil import rmtree
import pytest
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
@pytest.mark.skipif(sys.version_info >= (3, 8), reason='multiprocessing issues in Python 3.8')
def test_run_multiproc_nondaemon_false():
    """
    This is the entry point for the test. Two times a pipe of several
    multiprocessing jobs gets executed. First, without the nondaemon flag.
    Second, with the nondaemon flag.

    Since the processes of the pipe start child processes, the execution only
    succeeds when the non_daemon flag is on.
    """
    shouldHaveFailed = False
    try:
        run_multiproc_nondaemon_with_flag(False)
    except:
        shouldHaveFailed = True
    assert shouldHaveFailed