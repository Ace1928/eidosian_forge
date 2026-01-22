import atexit
import os
import signal
import sys
import typing as t
import uuid
import warnings
from jupyter_core.application import base_aliases, base_flags
from traitlets import CBool, CUnicode, Dict, List, Type, Unicode
from traitlets.config.application import boolean_flag
from . import KernelManager, connect, find_connection_file, tunnel_to_kernel
from .blocking import BlockingKernelClient
from .connect import KernelConnectionInfo
from .kernelspec import NoSuchKernel
from .localinterfaces import localhost
from .restarter import KernelRestarter
from .session import Session
from .utils import _filefind
class IPythonConsoleApp(JupyterConsoleApp):
    """An app to manage an ipython console."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialize the app."""
        warnings.warn('IPythonConsoleApp is deprecated. Use JupyterConsoleApp', stacklevel=2)
        super().__init__(*args, **kwargs)