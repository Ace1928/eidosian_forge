import asyncio
from contextlib import suppress
from typing import Any, Optional, Tuple
from .base_protocol import BaseProtocol
from .client_exceptions import (
from .helpers import BaseTimerContext, status_code_must_be_empty_body
from .http import HttpResponseParser, RawResponseMessage
from .streams import EMPTY_PAYLOAD, DataQueue, StreamReader
def _on_read_timeout(self) -> None:
    exc = ServerTimeoutError('Timeout on reading data from socket')
    self.set_exception(exc)
    if self._payload is not None:
        self._payload.set_exception(exc)