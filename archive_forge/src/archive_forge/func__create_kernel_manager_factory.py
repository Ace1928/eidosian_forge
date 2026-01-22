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
def _create_kernel_manager_factory(self) -> t.Callable:
    kernel_manager_ctor = import_item(self.kernel_manager_class)

    def create_kernel_manager(*args: t.Any, **kwargs: t.Any) -> KernelManager:
        if self.shared_context:
            if self.context.closed:
                self.context = self._context_default()
            kwargs.setdefault('context', self.context)
        km = kernel_manager_ctor(*args, **kwargs)
        return km
    return create_kernel_manager