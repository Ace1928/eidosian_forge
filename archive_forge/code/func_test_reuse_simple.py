from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_reuse_simple() -> None:
    p = ConnectionPair()
    p.send(CLIENT, [Request(method='GET', target='/', headers=[('Host', 'a')]), EndOfMessage()])
    p.send(SERVER, [Response(status_code=200, headers=[(b'transfer-encoding', b'chunked')]), EndOfMessage()])
    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: DONE}
        conn.start_next_cycle()
    p.send(CLIENT, [Request(method='DELETE', target='/foo', headers=[('Host', 'a')]), EndOfMessage()])
    p.send(SERVER, [Response(status_code=404, headers=[(b'transfer-encoding', b'chunked')]), EndOfMessage()])