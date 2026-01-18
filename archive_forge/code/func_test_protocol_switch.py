from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_protocol_switch() -> None:
    for req, deny, accept in [(Request(method='CONNECT', target='example.com:443', headers=[('Host', 'foo'), ('Content-Length', '1')]), Response(status_code=404, headers=[(b'transfer-encoding', b'chunked')]), Response(status_code=200, headers=[(b'transfer-encoding', b'chunked')])), (Request(method='GET', target='/', headers=[('Host', 'foo'), ('Content-Length', '1'), ('Upgrade', 'a, b')]), Response(status_code=200, headers=[(b'transfer-encoding', b'chunked')]), InformationalResponse(status_code=101, headers=[('Upgrade', 'a')])), (Request(method='CONNECT', target='example.com:443', headers=[('Host', 'foo'), ('Content-Length', '1'), ('Upgrade', 'a, b')]), Response(status_code=404, headers=[(b'transfer-encoding', b'chunked')]), Response(status_code=200, headers=[(b'transfer-encoding', b'chunked')])), (Request(method='CONNECT', target='example.com:443', headers=[('Host', 'foo'), ('Content-Length', '1'), ('Upgrade', 'a, b')]), Response(status_code=404, headers=[(b'transfer-encoding', b'chunked')]), InformationalResponse(status_code=101, headers=[('Upgrade', 'b')]))]:

        def setup() -> ConnectionPair:
            p = ConnectionPair()
            p.send(CLIENT, req)
            for conn in p.conns:
                assert conn.states[CLIENT] is SEND_BODY
            p.send(CLIENT, [Data(data=b'1'), EndOfMessage()])
            for conn in p.conns:
                assert conn.states[CLIENT] is MIGHT_SWITCH_PROTOCOL
            assert p.conn[SERVER].next_event() is PAUSED
            return p
        p = setup()
        p.send(SERVER, deny)
        for conn in p.conns:
            assert conn.states == {CLIENT: DONE, SERVER: SEND_BODY}
        p.send(SERVER, EndOfMessage())
        for conn in p.conns:
            conn.start_next_cycle()
        p = setup()
        p.send(SERVER, accept)
        for conn in p.conns:
            assert conn.states == {CLIENT: SWITCHED_PROTOCOL, SERVER: SWITCHED_PROTOCOL}
            conn.receive_data(b'123')
            assert conn.next_event() is PAUSED
            conn.receive_data(b'456')
            assert conn.next_event() is PAUSED
            assert conn.trailing_data == (b'123456', False)
        p = setup()
        sc = p.conn[SERVER]
        sc.receive_data(b'GET / HTTP/1.0\r\n\r\n')
        assert sc.next_event() is PAUSED
        assert sc.trailing_data == (b'GET / HTTP/1.0\r\n\r\n', False)
        sc.send(deny)
        assert sc.next_event() is PAUSED
        sc.send(EndOfMessage())
        sc.start_next_cycle()
        assert get_all_events(sc) == [Request(method='GET', target='/', headers=[], http_version='1.0'), EndOfMessage()]
        p = setup()
        sc = p.conn[SERVER]
        sc.receive_data(b'')
        assert sc.next_event() is PAUSED
        assert sc.trailing_data == (b'', True)
        p.send(SERVER, accept)
        assert sc.next_event() is PAUSED
        p = setup()
        sc = p.conn[SERVER]
        sc.receive_data(b'')
        assert sc.next_event() is PAUSED
        sc.send(deny)
        assert sc.next_event() == ConnectionClosed()
        p = setup()
        with pytest.raises(LocalProtocolError):
            p.conn[CLIENT].send(Request(method='GET', target='/', headers=[('Host', 'a')]))
        p = setup()
        p.send(SERVER, accept)
        with pytest.raises(LocalProtocolError):
            p.conn[SERVER].send(Data(data=b'123'))