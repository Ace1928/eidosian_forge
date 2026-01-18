from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_special_exceptions_for_lost_connection_in_message_body() -> None:
    c = Connection(SERVER)
    c.receive_data(b'POST / HTTP/1.1\r\nHost: example.com\r\nContent-Length: 100\r\n\r\n')
    assert type(c.next_event()) is Request
    assert c.next_event() is NEED_DATA
    c.receive_data(b'12345')
    assert c.next_event() == Data(data=b'12345')
    c.receive_data(b'')
    with pytest.raises(RemoteProtocolError) as excinfo:
        c.next_event()
    assert 'received 5 bytes' in str(excinfo.value)
    assert 'expected 100' in str(excinfo.value)
    c = Connection(SERVER)
    c.receive_data(b'POST / HTTP/1.1\r\nHost: example.com\r\nTransfer-Encoding: chunked\r\n\r\n')
    assert type(c.next_event()) is Request
    assert c.next_event() is NEED_DATA
    c.receive_data(b'8\r\n012345')
    assert c.next_event().data == b'012345'
    c.receive_data(b'')
    with pytest.raises(RemoteProtocolError) as excinfo:
        c.next_event()
    assert 'incomplete chunked read' in str(excinfo.value)