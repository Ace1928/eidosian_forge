from typing import Any, Callable, Generator, List
import pytest
from .._events import (
from .._headers import Headers, normalize_and_validate
from .._readers import (
from .._receivebuffer import ReceiveBuffer
from .._state import (
from .._util import LocalProtocolError
from .._writers import (
from .helpers import normalize_data_events
def test_writers_simple() -> None:
    for (role, state), event, binary in SIMPLE_CASES:
        tw(WRITERS[role, state], event, binary)