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
def _send_stream_data(self, request: Request, stream_id: int, data: bytes) -> None:
    """
        Send a single chunk of data in one or more data frames.
        """
    while data:
        max_flow = self._wait_for_outgoing_flow(request, stream_id)
        chunk_size = min(len(data), max_flow)
        chunk, data = (data[:chunk_size], data[chunk_size:])
        self._h2_state.send_data(stream_id, chunk)
        self._write_outgoing_data(request)