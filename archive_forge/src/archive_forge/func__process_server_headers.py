import abc
import asyncio
import base64
import hashlib
import os
import sys
import struct
import tornado
from urllib.parse import urlparse
import warnings
import zlib
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen, httpclient, httputil
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError, IOStream
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado import simple_httpclient
from tornado.queues import Queue
from tornado.tcpclient import TCPClient
from tornado.util import _websocket_mask
from typing import (
from types import TracebackType
def _process_server_headers(self, key: Union[str, bytes], headers: httputil.HTTPHeaders) -> None:
    """Process the headers sent by the server to this client connection.

        'key' is the websocket handshake challenge/response key.
        """
    assert headers['Upgrade'].lower() == 'websocket'
    assert headers['Connection'].lower() == 'upgrade'
    accept = self.compute_accept_value(key)
    assert headers['Sec-Websocket-Accept'] == accept
    extensions = self._parse_extensions_header(headers)
    for ext in extensions:
        if ext[0] == 'permessage-deflate' and self._compression_options is not None:
            self._create_compressors('client', ext[1])
        else:
            raise ValueError('unsupported extension %r', ext)
    self.selected_subprotocol = headers.get('Sec-WebSocket-Protocol', None)