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
def _impl_test_interactive_timers():
    import os
    from unittest.mock import Mock
    import matplotlib.pyplot as plt
    pause_time = 2 if os.getenv('CI') else 0.5
    fig = plt.figure()
    plt.pause(pause_time)
    timer = fig.canvas.new_timer(0.1)
    mock = Mock()
    timer.add_callback(mock)
    timer.start()
    plt.pause(pause_time)
    timer.stop()
    assert mock.call_count > 1
    mock.call_count = 0
    timer.single_shot = True
    timer.start()
    plt.pause(pause_time)
    assert mock.call_count == 1
    timer.start()
    plt.pause(pause_time)
    assert mock.call_count == 2
    plt.close('all')