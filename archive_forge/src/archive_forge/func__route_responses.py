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
def _route_responses(self):
    """
        Reads responses from the websocket and routes each to the appropriate channel queue based
        on the message's channel.  It does this for the duration of the class's lifetime until the
        channels are stopped, at which time the socket is closed (unblocking the router) and
        the thread terminates.  If shutdown happens to occur while processing a response (unlikely),
        termination takes place via the loop control boolean.
        """
    try:
        while not self._channels_stopped:
            assert self.channel_socket is not None
            raw_message = self.channel_socket.recv()
            if not raw_message:
                break
            response_message = json_decode(utf8(raw_message))
            channel = response_message['channel']
            assert self._channel_queues is not None
            self._channel_queues[channel].put_nowait(response_message)
    except websocket.WebSocketConnectionClosedException:
        pass
    except BaseException as be:
        if not self._channels_stopped:
            self.log.warning(f'Unexpected exception encountered ({be})')
    assert self._channel_queues is not None
    for channel_queue in self._channel_queues.values():
        channel_queue.response_router_finished = True
    self.log.debug('Response router thread exiting...')