import asyncio
import functools
import os
import re
import signal
import sys
import typing as t
import uuid
import warnings
from asyncio.futures import Future
from concurrent.futures import Future as CFuture
from contextlib import contextmanager
from enum import Enum
import zmq
from jupyter_core.utils import run_sync
from traitlets import (
from traitlets.utils.importstring import import_item
from . import kernelspec
from .asynchronous import AsyncKernelClient
from .blocking import BlockingKernelClient
from .client import KernelClient
from .connect import ConnectionFileMixin
from .managerabc import KernelManagerABC
from .provisioning import KernelProvisionerBase
from .provisioning import KernelProvisionerFactory as KPF  # noqa
@observe('kernel_name')
def _kernel_name_changed(self, change: t.Dict[str, str]) -> None:
    self._kernel_spec = None
    if change['new'] == 'python':
        self.kernel_name = kernelspec.NATIVE_KERNEL_NAME