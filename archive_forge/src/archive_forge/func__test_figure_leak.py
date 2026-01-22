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
def _test_figure_leak():
    import gc
    import sys
    import psutil
    from matplotlib import pyplot as plt
    t = float(sys.argv[1])
    p = psutil.Process()
    for _ in range(2):
        fig = plt.figure()
        if t:
            plt.pause(t)
        plt.close(fig)
    mem = p.memory_info().rss
    gc.collect()
    for _ in range(5):
        fig = plt.figure()
        if t:
            plt.pause(t)
        plt.close(fig)
        gc.collect()
    growth = p.memory_info().rss - mem
    print(growth)