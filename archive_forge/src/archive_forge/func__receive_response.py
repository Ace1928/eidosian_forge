import enum
import logging
import time
import types
import typing
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, Semaphore, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
def _receive_response(self, request: Request, stream_id: int) -> typing.Tuple[int, typing.List[typing.Tuple[bytes, bytes]]]:
    """
        Return the response status code and headers for a given stream ID.
        """
    while True:
        event = self._receive_stream_event(request, stream_id)
        if isinstance(event, h2.events.ResponseReceived):
            break
    status_code = 200
    headers = []
    for k, v in event.headers:
        if k == b':status':
            status_code = int(v.decode('ascii', errors='ignore'))
        elif not k.startswith(b':'):
            headers.append((k, v))
    return (status_code, headers)