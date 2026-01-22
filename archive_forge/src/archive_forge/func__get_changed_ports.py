from __future__ import annotations
import asyncio
import os
import pathlib
import typing as t
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial, wraps
from jupyter_client.ioloop.manager import AsyncIOLoopKernelManager
from jupyter_client.multikernelmanager import AsyncMultiKernelManager, MultiKernelManager
from jupyter_client.session import Session
from jupyter_core.paths import exists
from jupyter_core.utils import ensure_async
from jupyter_events import EventLogger
from jupyter_events.schema_registry import SchemaRegistryException
from overrides import overrides
from tornado import web
from tornado.concurrent import Future
from tornado.ioloop import IOLoop, PeriodicCallback
from traitlets import (
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH
from jupyter_server._tz import isoformat, utcnow
from jupyter_server.prometheus.metrics import KERNEL_CURRENTLY_RUNNING_TOTAL
from jupyter_server.utils import ApiPath, import_item, to_os_path
def _get_changed_ports(self, kernel_id):
    """Internal method to test if a kernel's ports have changed and, if so, return their values.

        This method does NOT update the captured ports for the kernel as that can only be done
        by ZMQChannelsHandler, but instead returns the new list of ports if they are different
        than those captured at startup.  This enables the ability to conditionally restart
        activity monitoring immediately following a kernel's restart (if ports have changed).
        """
    km = self.get_kernel(kernel_id)
    assert isinstance(km.ports, list)
    assert isinstance(self._kernel_ports[kernel_id], list)
    if km.ports != self._kernel_ports[kernel_id]:
        return km.ports
    return None