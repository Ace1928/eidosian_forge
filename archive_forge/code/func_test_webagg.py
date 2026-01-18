import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image
import pytest
import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper
@pytest.mark.skipif('TF_BUILD' in os.environ, reason='this test fails an azure for unknown reasons')
@pytest.mark.skipif(sys.platform == 'win32', reason='Cannot send SIGINT on Windows.')
def test_webagg():
    pytest.importorskip('tornado')
    proc = subprocess.Popen([sys.executable, '-c', inspect.getsource(_test_interactive_impl) + '\n_test_interactive_impl()', '{}'], env={**os.environ, 'MPLBACKEND': 'webagg', 'SOURCE_DATE_EPOCH': '0'})
    url = f'http://{mpl.rcParams['webagg.address']}:{mpl.rcParams['webagg.port']}'
    timeout = time.perf_counter() + _test_timeout
    try:
        while True:
            try:
                retcode = proc.poll()
                assert retcode is None
                conn = urllib.request.urlopen(url)
                break
            except urllib.error.URLError:
                if time.perf_counter() > timeout:
                    pytest.fail('Failed to connect to the webagg server.')
                else:
                    continue
        conn.close()
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=_test_timeout) == 0
    finally:
        if proc.poll() is None:
            proc.kill()