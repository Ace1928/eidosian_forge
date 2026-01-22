import ast
import copy
import importlib
import inspect
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from weakref import WeakKeyDictionary
import param
from bokeh.core.has_props import _default_resolver
from bokeh.document import Document
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from pyviz_comms import (
from .io.logging import panel_log_handler
from .io.state import state
from .util import param_watchers
def _detect_comms(self, params):
    called_before = self._comms_detected_before
    self._comms_detected_before = True
    if 'comms' in params:
        config.comms = params.pop('comms')
        return
    if called_before:
        return
    if 'google.colab' in sys.modules:
        try:
            import jupyter_bokeh
            config.comms = 'colab'
        except Exception:
            warnings.warn('Using Panel interactively in Colab notebooks requires the jupyter_bokeh package to be installed. Install it with:\n\n    !pip install jupyter_bokeh\n\nand try again.', stacklevel=5)
        return
    if 'VSCODE_CWD' in os.environ or 'VSCODE_PID' in os.environ:
        try:
            import jupyter_bokeh
            config.comms = 'vscode'
        except Exception:
            warnings.warn('Using Panel interactively in VSCode notebooks requires the jupyter_bokeh package to be installed. You can install it with:\n\n   pip install jupyter_bokeh\n\nor:\n    conda install jupyter_bokeh\n\nand try again.', stacklevel=5)
        self._ignore_bokeh_warnings()
        return