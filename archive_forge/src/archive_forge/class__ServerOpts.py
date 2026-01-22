from __future__ import annotations
import logging # isort:skip
import atexit
import signal
import socket
import sys
from types import FrameType
from typing import TYPE_CHECKING, Any, Mapping
from tornado import netutil, version as tornado_version
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from .. import __version__
from ..core import properties as p
from ..core.properties import (
from ..resources import DEFAULT_SERVER_PORT, server_url
from ..util.options import Options
from .tornado import DEFAULT_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES, BokehTornado
from .util import bind_sockets, create_hosts_allowlist
class _ServerOpts(Options):
    num_procs: int = Int(default=1, help='\n    The number of worker processes to start for the HTTP server. If an explicit\n    ``io_loop`` is also configured, then ``num_procs=1`` is the only compatible\n    value. Use ``BaseServer`` to coordinate an explicit ``IOLoop`` with a\n    multi-process HTTP server.\n\n    A value of 0 will auto detect number of cores.\n\n    Note that due to limitations inherent in Tornado, Windows does not support\n    ``num_procs`` values greater than one! In this case consider running\n    multiple Bokeh server instances behind a load balancer.\n    ')
    address: str | None = Nullable(String, help='\n    The address the server should listen on for HTTP requests.\n    ')
    port: int = Int(default=DEFAULT_SERVER_PORT, help='\n    The port number the server should listen on for HTTP requests.\n    ')
    unix_socket: str | None = Nullable(String, help='\n    The unix socket the server should bind to. Other network args\n    such as port, address, ssl options etc are incompatible with unix sockets.\n    Unix socket support is not available on windows.\n    ')
    prefix: str = String(default='', help='\n    A URL prefix to use for all Bokeh server paths.\n    ')
    index: str | None = Nullable(String, help='\n    A path to a Jinja2 template to use for the index "/"\n    ')
    allow_websocket_origin: list[str] | None = Nullable(p.List(String), help='\n    A list of hosts that can connect to the websocket.\n\n    This is typically required when embedding a Bokeh server app in an external\n    web site using :func:`~bokeh.embed.server_document` or similar.\n\n    If None, "localhost" is used.\n    ')
    use_xheaders: bool = Bool(default=False, help='\n    Whether to have the Bokeh server override the remote IP and URI scheme\n    and protocol for all requests with ``X-Real-Ip``, ``X-Forwarded-For``,\n    ``X-Scheme``, ``X-Forwarded-Proto`` headers (if they are provided).\n    ')
    ssl_certfile: str | None = Nullable(String, help='\n    The path to a certificate file for SSL termination.\n    ')
    ssl_keyfile: str | None = Nullable(String, help='\n    The path to a private key file for SSL termination.\n    ')
    ssl_password: str | None = Nullable(String, help='\n    A password to decrypt the SSL keyfile, if necessary.\n    ')
    websocket_max_message_size: int = Int(default=DEFAULT_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES, help='\n    Set the Tornado ``websocket_max_message_size`` value.\n    ')