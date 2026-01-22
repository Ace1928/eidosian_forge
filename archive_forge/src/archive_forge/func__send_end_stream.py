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
def _send_end_stream(self, request: Request, stream_id: int) -> None:
    """
        Send an empty data frame on on a given stream ID with the END_STREAM flag set.
        """
    self._h2_state.end_stream(stream_id)
    self._write_outgoing_data(request)