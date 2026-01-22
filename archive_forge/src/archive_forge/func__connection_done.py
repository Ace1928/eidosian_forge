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
def _connection_done(self, fut):
    """Handle a finished connection."""
    if not self.disconnected and fut.exception() is None:
        self.ws = fut.result()
        self.retry = 0
        self.log.debug(f'Connection is ready: ws: {self.ws}')
    else:
        self.log.warning(f"Websocket connection has been closed via client disconnect or due to error.  Kernel with ID '{self.kernel_id}' may not be terminated on GatewayClient: {GatewayClient.instance().url}")