from __future__ import annotations
import asyncio
import logging
import mimetypes
import os
import random
import warnings
from typing import Any, Optional, cast
from jupyter_client.session import Session
from tornado import web
from tornado.concurrent import Future
from tornado.escape import json_decode, url_escape, utf8
from tornado.httpclient import HTTPRequest
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.websocket import WebSocketHandler, websocket_connect
from traitlets.config.configurable import LoggingConfigurable
from ..base.handlers import APIHandler, JupyterHandler
from ..utils import url_path_join
from .gateway_client import GatewayClient
from ..services.kernels.handlers import _kernel_id_regex
from ..services.kernelspecs.handlers import kernel_name_regex
class GatewayWebSocketClient(LoggingConfigurable):
    """Proxy web socket connection to a kernel/enterprise gateway."""

    def __init__(self, **kwargs):
        """Initialize the gateway web socket client."""
        super().__init__()
        self.kernel_id = None
        self.ws = None
        self.ws_future: Future[Any] = Future()
        self.disconnected = False
        self.retry = 0

    async def _connect(self, kernel_id, message_callback):
        """Connect to the socket."""
        self.ws = None
        self.kernel_id = kernel_id
        client = GatewayClient.instance()
        assert client.ws_url is not None
        ws_url = url_path_join(client.ws_url, client.kernels_endpoint, url_escape(kernel_id), 'channels')
        self.log.info(f'Connecting to {ws_url}')
        kwargs: dict[str, Any] = {}
        kwargs = client.load_connection_args(**kwargs)
        request = HTTPRequest(ws_url, **kwargs)
        self.ws_future = cast('Future[Any]', websocket_connect(request))
        self.ws_future.add_done_callback(self._connection_done)
        loop = IOLoop.current()
        loop.add_future(self.ws_future, lambda future: self._read_messages(message_callback))

    def _connection_done(self, fut):
        """Handle a finished connection."""
        if not self.disconnected and fut.exception() is None:
            self.ws = fut.result()
            self.retry = 0
            self.log.debug(f'Connection is ready: ws: {self.ws}')
        else:
            self.log.warning(f"Websocket connection has been closed via client disconnect or due to error.  Kernel with ID '{self.kernel_id}' may not be terminated on GatewayClient: {GatewayClient.instance().url}")

    def _disconnect(self):
        """Handle a disconnect."""
        self.disconnected = True
        if self.ws is not None:
            self.ws.close()
        elif not self.ws_future.done():
            self.ws_future.cancel()
            self.log.debug(f'_disconnect: future cancelled, disconnected: {self.disconnected}')

    async def _read_messages(self, callback):
        """Read messages from gateway server."""
        while self.ws is not None:
            message = None
            if not self.disconnected:
                try:
                    message = await self.ws.read_message()
                except Exception as e:
                    self.log.error(f'Exception reading message from websocket: {e}')
                if message is None:
                    if not self.disconnected:
                        self.log.warning(f'Lost connection to Gateway: {self.kernel_id}')
                    break
                callback(message)
            else:
                break
        if not self.disconnected and self.retry < GatewayClient.instance().gateway_retry_max:
            jitter = random.randint(10, 100) * 0.01
            retry_interval = min(GatewayClient.instance().gateway_retry_interval * 2 ** self.retry, GatewayClient.instance().gateway_retry_interval_max) + jitter
            self.retry += 1
            self.log.info('Attempting to re-establish the connection to Gateway in %s secs (%s/%s): %s', retry_interval, self.retry, GatewayClient.instance().gateway_retry_max, self.kernel_id)
            await asyncio.sleep(retry_interval)
            loop = IOLoop.current()
            loop.spawn_callback(self._connect, self.kernel_id, callback)

    def on_open(self, kernel_id, message_callback, **kwargs):
        """Web socket connection open against gateway server."""
        loop = IOLoop.current()
        loop.spawn_callback(self._connect, kernel_id, message_callback)

    def on_message(self, message):
        """Send message to gateway server."""
        if self.ws is None:
            loop = IOLoop.current()
            loop.add_future(self.ws_future, lambda future: self._write_message(message))
        else:
            self._write_message(message)

    def _write_message(self, message):
        """Send message to gateway server."""
        try:
            if not self.disconnected and self.ws is not None:
                self.ws.write_message(message)
        except Exception as e:
            self.log.error(f'Exception writing message to websocket: {e}')

    def on_close(self):
        """Web socket closed event."""
        self._disconnect()