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
def _receive_response_body(self, request: Request, stream_id: int) -> typing.Iterator[bytes]:
    """
        Iterator that returns the bytes of the response body for a given stream ID.
        """
    while True:
        event = self._receive_stream_event(request, stream_id)
        if isinstance(event, h2.events.DataReceived):
            amount = event.flow_controlled_length
            self._h2_state.acknowledge_received_data(amount, stream_id)
            self._write_outgoing_data(request)
            yield event.data
        elif isinstance(event, h2.events.StreamEnded):
            break