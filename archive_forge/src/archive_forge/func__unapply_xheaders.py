import socket
import ssl
from tornado.escape import native_str
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
from tornado import httputil
from tornado import iostream
from tornado import netutil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
import typing
from typing import Union, Any, Dict, Callable, List, Type, Tuple, Optional, Awaitable
def _unapply_xheaders(self) -> None:
    """Undo changes from `_apply_xheaders`.

        Xheaders are per-request so they should not leak to the next
        request on the same connection.
        """
    self.remote_ip = self._orig_remote_ip
    self.protocol = self._orig_protocol