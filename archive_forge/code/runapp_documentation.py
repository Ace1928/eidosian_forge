from __future__ import annotations
import queue
import signal
import sys
import time
import typing as t
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Any, Dict, Float
from traitlets.config import catch_config_error
from . import __version__
from .consoleapp import JupyterConsoleApp, app_aliases, app_flags
Start the application.