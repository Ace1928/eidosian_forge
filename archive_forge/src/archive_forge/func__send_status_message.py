from __future__ import annotations
import asyncio
import json
import time
import typing as t
import weakref
from concurrent.futures import Future
from textwrap import dedent
from jupyter_client import protocol_version as client_protocol_version  # type:ignore[attr-defined]
from tornado import gen, web
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketClosedError
from traitlets import Any, Bool, Dict, Float, Instance, Int, List, Unicode, default
from jupyter_core.utils import ensure_async
from jupyter_server.transutils import _i18n
from ..websocket import KernelWebsocketHandler
from .abc import KernelWebsocketConnectionABC
from .base import (
def _send_status_message(self, status):
    """Send a status message."""
    iopub = self.channels.get('iopub', None)
    if iopub and (not iopub.closed()):
        iopub.flush()
    msg = self.session.msg('status', {'execution_state': status})
    if self.subprotocol == 'v1.kernel.websocket.jupyter.org':
        bin_msg = serialize_msg_to_ws_v1(msg, 'iopub', self.session.pack)
        self.write_message(bin_msg, binary=True)
    else:
        msg['channel'] = 'iopub'
        self.write_message(json.dumps(msg, default=json_default))