from __future__ import annotations
import asyncio
import json
import os
import socket
import typing as t
import uuid
from functools import wraps
from pathlib import Path
import zmq
from traitlets import Any, Bool, Dict, DottedObjectName, Instance, Unicode, default, observe
from traitlets.config.configurable import LoggingConfigurable
from traitlets.utils.importstring import import_item
from .connect import KernelConnectionInfo
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
from .utils import ensure_async, run_sync, utcnow
def _using_pending_kernels(self) -> bool:
    """Returns a boolean; a clearer method for determining if
        this multikernelmanager is using pending kernels or not
        """
    return getattr(self, 'use_pending_kernels', False)