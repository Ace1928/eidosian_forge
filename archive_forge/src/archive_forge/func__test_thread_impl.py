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
def _test_thread_impl():
    from concurrent.futures import ThreadPoolExecutor
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    mpl.rcParams.update({'webagg.open_in_browser': False, 'webagg.port_retries': 1})
    fig, ax = plt.subplots()
    plt.pause(0.5)
    future = ThreadPoolExecutor().submit(ax.plot, [1, 3, 6])
    future.result()
    fig.canvas.mpl_connect('close_event', print)
    future = ThreadPoolExecutor().submit(fig.canvas.draw)
    plt.pause(0.5)
    future.result()
    plt.close()
    if plt.rcParams['backend'].startswith('WX'):
        fig.canvas.flush_events()