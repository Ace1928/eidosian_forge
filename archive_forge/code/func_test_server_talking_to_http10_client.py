from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_server_talking_to_http10_client() -> None:
    c = Connection(SERVER)
    assert receive_and_get(c, b'GET / HTTP/1.0\r\n\r\n') == [Request(method='GET', target='/', headers=[], http_version='1.0'), EndOfMessage()]
    assert c.their_state is MUST_CLOSE
    assert c.send(Response(status_code=200, headers=[])) == b'HTTP/1.1 200 \r\nConnection: close\r\n\r\n'
    assert c.send(Data(data=b'12345')) == b'12345'
    assert c.send(EndOfMessage()) == b''
    assert c.our_state is MUST_CLOSE
    c = Connection(SERVER)
    assert receive_and_get(c, b'POST / HTTP/1.0\r\nContent-Length: 10\r\n\r\n1') == [Request(method='POST', target='/', headers=[('Content-Length', '10')], http_version='1.0'), Data(data=b'1')]
    assert receive_and_get(c, b'234567890') == [Data(data=b'234567890'), EndOfMessage()]
    assert c.their_state is MUST_CLOSE
    assert receive_and_get(c, b'') == [ConnectionClosed()]