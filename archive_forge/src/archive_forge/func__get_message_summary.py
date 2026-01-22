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
@staticmethod
def _get_message_summary(message):
    """Get a summary of a message."""
    summary = []
    message_type = message['msg_type']
    summary.append(f'type: {message_type}')
    if message_type == 'status':
        summary.append(', state: {}'.format(message['content']['execution_state']))
    elif message_type == 'error':
        summary.append(', {}:{}:{}'.format(message['content']['ename'], message['content']['evalue'], message['content']['traceback']))
    else:
        summary.append(', ...')
    return ''.join(summary)