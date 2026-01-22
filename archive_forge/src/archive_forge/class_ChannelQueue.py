from __future__ import annotations
import asyncio
import datetime
import json
import os
from logging import Logger
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any, Optional, cast
import websocket
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_core.utils import ensure_async
from tornado import web
from tornado.escape import json_decode, json_encode, url_escape, utf8
from traitlets import DottedObjectName, Instance, Type, default
from .._tz import UTC, utcnow
from ..services.kernels.kernelmanager import (
from ..services.sessions.sessionmanager import SessionManager
from ..utils import url_path_join
from .gateway_client import GatewayClient, gateway_request
class ChannelQueue(Queue):
    """A queue for a named channel."""
    channel_name: Optional[str] = None
    response_router_finished: bool

    def __init__(self, channel_name: str, channel_socket: websocket.WebSocket, log: Logger):
        """Initialize a channel queue."""
        super().__init__()
        self.channel_name = channel_name
        self.channel_socket = channel_socket
        self.log = log
        self.response_router_finished = False

    async def _async_get(self, timeout=None):
        """Asynchronously get from the queue."""
        if timeout is None:
            timeout = float('inf')
        elif timeout < 0:
            msg = "'timeout' must be a non-negative number"
            raise ValueError(msg)
        end_time = monotonic() + timeout
        while True:
            try:
                return self.get(block=False)
            except Empty:
                if self.response_router_finished:
                    msg = 'Response router had finished'
                    raise RuntimeError(msg) from None
                if monotonic() > end_time:
                    raise
                await asyncio.sleep(0)

    async def get_msg(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Get a message from the queue."""
        timeout = kwargs.get('timeout', 1)
        msg = await self._async_get(timeout=timeout)
        self.log.debug('Received message on channel: {}, msg_id: {}, msg_type: {}'.format(self.channel_name, msg['msg_id'], msg['msg_type'] if msg else 'null'))
        self.task_done()
        return cast('dict[str, Any]', msg)

    def send(self, msg: dict[str, Any]) -> None:
        """Send a message to the queue."""
        message = json.dumps(msg, default=ChannelQueue.serialize_datetime).replace('</', '<\\/')
        self.log.debug('Sending message on channel: {}, msg_id: {}, msg_type: {}'.format(self.channel_name, msg['msg_id'], msg['msg_type'] if msg else 'null'))
        self.channel_socket.send(message)

    @staticmethod
    def serialize_datetime(dt):
        """Serialize a datetime object."""
        if isinstance(dt, datetime.datetime):
            return dt.timestamp()
        return None

    def start(self) -> None:
        """Start the queue."""

    def stop(self) -> None:
        """Stop the queue."""
        if not self.empty():
            msgs = []
            while self.qsize():
                msg = self.get_nowait()
                if msg['msg_type'] != 'status':
                    msgs.append(msg['msg_type'])
            if self.channel_name == 'iopub' and 'shutdown_reply' in msgs:
                return
            if len(msgs):
                self.log.warning(f"Stopping channel '{self.channel_name}' with {len(msgs)} unprocessed non-status messages: {msgs}.")

    def is_alive(self) -> bool:
        """Whether the queue is alive."""
        return self.channel_socket is not None