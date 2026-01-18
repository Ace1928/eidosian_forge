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
def test_reject_non_vchar_in_path() -> None:
    for bad_char in b'\x00 \x7f\xee':
        message = bytearray(b'HEAD /')
        message.append(bad_char)
        message.extend(b' HTTP/1.1\r\nHost: foobar\r\n\r\n')
        with pytest.raises(LocalProtocolError):
            tr(READERS[CLIENT, IDLE], message, None)