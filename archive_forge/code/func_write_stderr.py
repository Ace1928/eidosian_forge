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
def write_stderr(self, error_message, parent_header):
    """Write a message to stderr."""
    self.log.warning(error_message)
    err_msg = self.session.msg('stream', content={'text': error_message + '\n', 'name': 'stderr'}, parent=parent_header)
    if self.subprotocol == 'v1.kernel.websocket.jupyter.org':
        bin_msg = serialize_msg_to_ws_v1(err_msg, 'iopub', self.session.pack)
        self.write_message(bin_msg, binary=True)
    else:
        err_msg['channel'] = 'iopub'
        self.write_message(json.dumps(err_msg, default=json_default))