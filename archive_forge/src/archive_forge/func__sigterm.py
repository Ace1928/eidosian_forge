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
def _sigterm(self, signum: int, frame: FrameType | None) -> None:
    print(f'Received signal {signum}, shutting down')
    self._loop.add_callback_from_signal(self._loop.stop)