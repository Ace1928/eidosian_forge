import logging
import sys
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShellABC
from traitlets import Any, Enum, Instance, List, Type, default
from ipykernel.ipkernel import IPythonKernel
from ipykernel.jsonutil import json_clean
from ipykernel.zmqshell import ZMQInteractiveShell
from ..iostream import BackgroundSocket, IOPubThread, OutStream
from .constants import INPROCESS_KEY
from .socket import DummySocket
@contextmanager
def _redirected_io(self):
    """Temporarily redirect IO to the kernel."""
    sys_stdout, sys_stderr = (sys.stdout, sys.stderr)
    try:
        sys.stdout, sys.stderr = (self.stdout, self.stderr)
        yield
    finally:
        sys.stdout, sys.stderr = (sys_stdout, sys_stderr)