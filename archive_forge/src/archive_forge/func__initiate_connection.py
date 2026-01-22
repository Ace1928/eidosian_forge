from collections import deque
from typing import (
import h11
from .connection import Connection, ConnectionState, ConnectionType
from .events import AcceptConnection, Event, RejectConnection, RejectData, Request
from .extensions import Extension
from .typing import Headers
from .utilities import (
def _initiate_connection(self, request: Request) -> bytes:
    self._initiating_request = request
    self._nonce = generate_nonce()
    headers = [(b'Host', request.host.encode('idna')), (b'Upgrade', b'WebSocket'), (b'Connection', b'Upgrade'), (b'Sec-WebSocket-Key', self._nonce), (b'Sec-WebSocket-Version', WEBSOCKET_VERSION)]
    if request.subprotocols:
        headers.append((b'Sec-WebSocket-Protocol', ', '.join(request.subprotocols).encode('ascii')))
    if request.extensions:
        offers: Dict[str, Union[str, bool]] = {}
        for e in request.extensions:
            assert isinstance(e, Extension)
            offers[e.name] = e.offer()
        extensions = []
        for name, params in offers.items():
            bname = name.encode('ascii')
            if isinstance(params, bool):
                if params:
                    extensions.append(bname)
            else:
                extensions.append(b'%s; %s' % (bname, params.encode('ascii')))
        if extensions:
            headers.append((b'Sec-WebSocket-Extensions', b', '.join(extensions)))
    upgrade = h11.Request(method=b'GET', target=request.target.encode('ascii'), headers=headers + request.extra_headers)
    return self._h11_connection.send(upgrade) or b''