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
Start the application.