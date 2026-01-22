import asyncio
import base64
import binascii
import hashlib
import json
import sys
from typing import Any, Final, Iterable, Optional, Tuple, cast
import attr
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import call_later, set_result
from .http import (
from .log import ws_logger
from .streams import EofStream, FlowControlDataQueue
from .typedefs import JSONDecoder, JSONEncoder
from .web_exceptions import HTTPBadRequest, HTTPException
from .web_request import BaseRequest
from .web_response import StreamResponse
def _pre_start(self, request: BaseRequest) -> Tuple[str, WebSocketWriter]:
    self._loop = request._loop
    headers, protocol, compress, notakeover = self._handshake(request)
    self.set_status(101)
    self.headers.update(headers)
    self.force_close()
    self._compress = compress
    transport = request._protocol.transport
    assert transport is not None
    writer = WebSocketWriter(request._protocol, transport, compress=compress, notakeover=notakeover)
    return (protocol, writer)