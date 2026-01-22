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
def _test_toolbar_button_la_mode_icon(fig):
    with tempfile.TemporaryDirectory() as tempdir:
        img = Image.new('LA', (26, 26))
        tmp_img_path = os.path.join(tempdir, 'test_la_icon.png')
        img.save(tmp_img_path)

        class CustomTool(ToolToggleBase):
            image = tmp_img_path
            description = ''
        toolmanager = fig.canvas.manager.toolmanager
        toolbar = fig.canvas.manager.toolbar
        toolmanager.add_tool('test', CustomTool)
        toolbar.add_tool('test', 'group')