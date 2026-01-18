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
def unlisten(self) -> None:
    """ Stop listening on ports. The server will no longer be usable after
        calling this function.

        .. note::
            This function is mostly useful for tests

        Returns:
            None

        """
    self._http.stop()
    self.io_loop.add_callback(self._http.close_all_connections)