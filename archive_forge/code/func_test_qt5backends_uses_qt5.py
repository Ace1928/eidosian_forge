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
def test_qt5backends_uses_qt5():
    qt5_bindings = [dep for dep in ['PyQt5', 'pyside2'] if importlib.util.find_spec(dep) is not None]
    qt6_bindings = [dep for dep in ['PyQt6', 'pyside6'] if importlib.util.find_spec(dep) is not None]
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        pytest.skip('need both QT6 and QT5 bindings')
    _run_helper(_implqt5agg, timeout=_test_timeout)
    if importlib.util.find_spec('pycairo') is not None:
        _run_helper(_implcairo, timeout=_test_timeout)
    _run_helper(_implcore, timeout=_test_timeout)