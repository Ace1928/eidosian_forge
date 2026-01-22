import typing
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette._utils import collapse_excgroups
from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect, Request
from starlette.responses import ContentStream, Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
class _StreamingResponse(StreamingResponse):

    def __init__(self, content: ContentStream, status_code: int=200, headers: typing.Optional[typing.Mapping[str, str]]=None, media_type: typing.Optional[str]=None, background: typing.Optional[BackgroundTask]=None, info: typing.Optional[typing.Mapping[str, typing.Any]]=None) -> None:
        self._info = info
        super().__init__(content, status_code, headers, media_type, background)

    async def stream_response(self, send: Send) -> None:
        if self._info:
            await send({'type': 'http.response.debug', 'info': self._info})
        return await super().stream_response(send)