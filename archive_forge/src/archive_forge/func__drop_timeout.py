import asyncio
from contextlib import suppress
from typing import Any, Optional, Tuple
from .base_protocol import BaseProtocol
from .client_exceptions import (
from .helpers import BaseTimerContext, status_code_must_be_empty_body
from .http import HttpResponseParser, RawResponseMessage
from .streams import EMPTY_PAYLOAD, DataQueue, StreamReader
def _drop_timeout(self) -> None:
    if self._read_timeout_handle is not None:
        self._read_timeout_handle.cancel()
        self._read_timeout_handle = None