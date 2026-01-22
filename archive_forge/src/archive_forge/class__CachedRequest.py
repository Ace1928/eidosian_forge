import typing
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette._utils import collapse_excgroups
from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect, Request
from starlette.responses import ContentStream, Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
class _CachedRequest(Request):
    """
    If the user calls Request.body() from their dispatch function
    we cache the entire request body in memory and pass that to downstream middlewares,
    but if they call Request.stream() then all we do is send an
    empty body so that downstream things don't hang forever.
    """

    def __init__(self, scope: Scope, receive: Receive):
        super().__init__(scope, receive)
        self._wrapped_rcv_disconnected = False
        self._wrapped_rcv_consumed = False
        self._wrapped_rc_stream = self.stream()

    async def wrapped_receive(self) -> Message:
        if self._wrapped_rcv_disconnected:
            return {'type': 'http.disconnect'}
        if self._wrapped_rcv_consumed:
            if self._is_disconnected:
                self._wrapped_rcv_disconnected = True
                return {'type': 'http.disconnect'}
            msg = await self.receive()
            if msg['type'] != 'http.disconnect':
                raise RuntimeError(f'Unexpected message received: {msg['type']}')
            return msg
        if getattr(self, '_body', None) is not None:
            self._wrapped_rcv_consumed = True
            return {'type': 'http.request', 'body': self._body, 'more_body': False}
        elif self._stream_consumed:
            self._wrapped_rcv_consumed = True
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        else:
            try:
                stream = self.stream()
                chunk = await stream.__anext__()
                self._wrapped_rcv_consumed = self._stream_consumed
                return {'type': 'http.request', 'body': chunk, 'more_body': not self._stream_consumed}
            except ClientDisconnect:
                self._wrapped_rcv_disconnected = True
                return {'type': 'http.disconnect'}