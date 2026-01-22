from __future__ import annotations
import asyncio
import functools
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass as python_dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import (
from urllib.parse import urlparse
import anyio
import fastapi
import gradio_client.utils as client_utils
import httpx
import multipart
from gradio_client.documentation import document
from multipart.multipart import parse_options_header
from starlette.datastructures import FormData, Headers, MutableHeaders, UploadFile
from starlette.formparsers import MultiPartException, MultipartPart
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from gradio import processing_utils, utils
from gradio.data_classes import PredictBody
from gradio.exceptions import Error
from gradio.helpers import EventData
from gradio.state_holder import SessionState
class CustomCORSMiddleware:

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.all_methods = ('DELETE', 'GET', 'HEAD', 'OPTIONS', 'PATCH', 'POST', 'PUT')
        self.preflight_headers = {'Access-Control-Allow-Methods': ', '.join(self.all_methods), 'Access-Control-Max-Age': str(600)}
        self.simple_headers = {'Access-Control-Allow-Credentials': 'true'}
        self.localhost_aliases = ['localhost', '127.0.0.1', '0.0.0.0', 'null']

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get('origin')
        if origin is None:
            await self.app(scope, receive, send)
            return
        if scope['method'] == 'OPTIONS' and 'access-control-request-method' in headers:
            response = self.preflight_response(request_headers=headers)
            await response(scope, receive, send)
            return
        await self.simple_response(scope, receive, send, request_headers=headers)

    def preflight_response(self, request_headers: Headers) -> Response:
        headers = dict(self.preflight_headers)
        origin = request_headers['Origin']
        if self.is_valid_origin(request_headers):
            headers['Access-Control-Allow-Origin'] = origin
        requested_headers = request_headers.get('access-control-request-headers')
        if requested_headers is not None:
            headers['Access-Control-Allow-Headers'] = requested_headers
        return PlainTextResponse('OK', status_code=200, headers=headers)

    async def simple_response(self, scope: Scope, receive: Receive, send: Send, request_headers: Headers) -> None:
        send = functools.partial(self._send, send=send, request_headers=request_headers)
        await self.app(scope, receive, send)

    async def _send(self, message: Message, send: Send, request_headers: Headers) -> None:
        if message['type'] != 'http.response.start':
            await send(message)
            return
        message.setdefault('headers', [])
        headers = MutableHeaders(scope=message)
        headers.update(self.simple_headers)
        has_cookie = 'cookie' in request_headers
        origin = request_headers['Origin']
        if has_cookie or self.is_valid_origin(request_headers):
            self.allow_explicit_origin(headers, origin)
        await send(message)

    def is_valid_origin(self, request_headers: Headers) -> bool:
        origin = request_headers['Origin']
        host = request_headers['Host']
        host_name = get_hostname(host)
        origin_name = get_hostname(origin)
        return host_name not in self.localhost_aliases or origin_name in self.localhost_aliases

    @staticmethod
    def allow_explicit_origin(headers: MutableHeaders, origin: str) -> None:
        headers['Access-Control-Allow-Origin'] = origin
        headers.add_vary_header('Origin')