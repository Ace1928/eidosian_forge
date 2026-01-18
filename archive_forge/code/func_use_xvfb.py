import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
@use_xvfb.setter
def use_xvfb(self, val):
    valid_vals = [True, False, 'auto']
    if val is None:
        self._props.pop('use_xvfb', None)
    else:
        if val not in valid_vals:
            raise ValueError('\nThe use_xvfb property must be one of {valid_vals}\n    Received value of type {typ}: {val}'.format(valid_vals=valid_vals, typ=type(val), val=repr(val)))
        self._props['use_xvfb'] = val
    reset_status()