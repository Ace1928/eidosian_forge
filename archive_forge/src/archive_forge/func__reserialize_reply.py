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
def _reserialize_reply(self, msg_or_list, channel=None):
    """Reserialize a reply message using JSON.

        msg_or_list can be an already-deserialized msg dict or the zmq buffer list.
        If it is the zmq list, it will be deserialized with self.session.

        This takes the msg list from the ZMQ socket and serializes the result for the websocket.
        This method should be used by self._on_zmq_reply to build messages that can
        be sent back to the browser.

        """
    if isinstance(msg_or_list, dict):
        msg = msg_or_list
    else:
        _, msg_list = self.session.feed_identities(msg_or_list)
        msg = self.session.deserialize(msg_list)
    if channel:
        msg['channel'] = channel
    if msg['buffers']:
        buf = serialize_binary_message(msg)
        return buf
    else:
        return json.dumps(msg, default=json_default)