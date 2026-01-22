import asyncio
import os
import re
import signal
import sys
from types import FrameType
from typing import Any, Awaitable, Callable, Optional, Union  # noqa
from gunicorn.config import AccessLogFormat as GunicornAccessLogFormat
from gunicorn.workers import base
from aiohttp import web
from .helpers import set_result
from .web_app import Application
from .web_log import AccessLogger
@staticmethod
def _create_ssl_context(cfg: Any) -> 'SSLContext':
    """Creates SSLContext instance for usage in asyncio.create_server.

        See ssl.SSLSocket.__init__ for more details.
        """
    if ssl is None:
        raise RuntimeError('SSL is not supported.')
    ctx = ssl.SSLContext(cfg.ssl_version)
    ctx.load_cert_chain(cfg.certfile, cfg.keyfile)
    ctx.verify_mode = cfg.cert_reqs
    if cfg.ca_certs:
        ctx.load_verify_locations(cfg.ca_certs)
    if cfg.ciphers:
        ctx.set_ciphers(cfg.ciphers)
    return ctx