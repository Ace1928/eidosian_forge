from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_sendfile() -> None:

    class SendfilePlaceholder:

        def __len__(self) -> int:
            return 10
    placeholder = SendfilePlaceholder()

    def setup(header: Tuple[str, str], http_version: str) -> Tuple[Connection, Optional[List[bytes]]]:
        c = Connection(SERVER)
        receive_and_get(c, 'GET / HTTP/{}\r\nHost: a\r\n\r\n'.format(http_version).encode('ascii'))
        headers = []
        if header:
            headers.append(header)
        c.send(Response(status_code=200, headers=headers))
        return (c, c.send_with_data_passthrough(Data(data=placeholder)))
    c, data = setup(('Content-Length', '10'), '1.1')
    assert data == [placeholder]
    c.send(EndOfMessage())
    _, data = setup(('Transfer-Encoding', 'chunked'), '1.1')
    assert placeholder in data
    data[data.index(placeholder)] = b'x' * 10
    assert b''.join(data) == b'a\r\nxxxxxxxxxx\r\n'
    c, data = setup(None, '1.0')
    assert data == [placeholder]
    assert c.our_state is SEND_BODY