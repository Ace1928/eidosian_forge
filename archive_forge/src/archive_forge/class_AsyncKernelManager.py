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
class AsyncKernelManager(KernelManager):
    """An async kernel manager."""
    client_class: DottedObjectName = DottedObjectName('jupyter_client.asynchronous.AsyncKernelClient')
    client_factory: Type = Type(klass='jupyter_client.asynchronous.AsyncKernelClient')
    context: Instance = Instance(zmq.asyncio.Context)

    @default('context')
    def _context_default(self) -> zmq.asyncio.Context:
        self._created_context = True
        return zmq.asyncio.Context()

    def client(self, **kwargs: t.Any) -> AsyncKernelClient:
        """Get a client for the manager."""
        return super().client(**kwargs)
    _launch_kernel = KernelManager._async_launch_kernel
    start_kernel: t.Callable[..., t.Awaitable] = KernelManager._async_start_kernel
    pre_start_kernel: t.Callable[..., t.Awaitable] = KernelManager._async_pre_start_kernel
    post_start_kernel: t.Callable[..., t.Awaitable] = KernelManager._async_post_start_kernel
    request_shutdown: t.Callable[..., t.Awaitable] = KernelManager._async_request_shutdown
    finish_shutdown: t.Callable[..., t.Awaitable] = KernelManager._async_finish_shutdown
    cleanup_resources: t.Callable[..., t.Awaitable] = KernelManager._async_cleanup_resources
    shutdown_kernel: t.Callable[..., t.Awaitable] = KernelManager._async_shutdown_kernel
    restart_kernel: t.Callable[..., t.Awaitable] = KernelManager._async_restart_kernel
    _send_kernel_sigterm = KernelManager._async_send_kernel_sigterm
    _kill_kernel = KernelManager._async_kill_kernel
    interrupt_kernel: t.Callable[..., t.Awaitable] = KernelManager._async_interrupt_kernel
    signal_kernel: t.Callable[..., t.Awaitable] = KernelManager._async_signal_kernel
    is_alive: t.Callable[..., t.Awaitable] = KernelManager._async_is_alive