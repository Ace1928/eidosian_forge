import os
import signal
import typing as t
import uuid
from jupyter_core.application import JupyterApp, base_flags
from tornado.ioloop import IOLoop
from traitlets import Unicode
from . import __version__
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
def _record_started(self) -> None:
    """For tests, create a file to indicate that we've started

        Do not rely on this except in our own tests!
        """
    fn = os.environ.get('JUPYTER_CLIENT_TEST_RECORD_STARTUP_PRIVATE')
    if fn is not None:
        with open(fn, 'wb'):
            pass