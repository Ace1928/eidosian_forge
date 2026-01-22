import io
import math
import sys
import typing
import warnings
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette.types import Receive, Scope, Send
class WSGIResponder:
    stream_send: ObjectSendStream[typing.MutableMapping[str, typing.Any]]
    stream_receive: ObjectReceiveStream[typing.MutableMapping[str, typing.Any]]

    def __init__(self, app: typing.Callable[..., typing.Any], scope: Scope) -> None:
        self.app = app
        self.scope = scope
        self.status = None
        self.response_headers = None
        self.stream_send, self.stream_receive = anyio.create_memory_object_stream(math.inf)
        self.response_started = False
        self.exc_info: typing.Any = None

    async def __call__(self, receive: Receive, send: Send) -> None:
        body = b''
        more_body = True
        while more_body:
            message = await receive()
            body += message.get('body', b'')
            more_body = message.get('more_body', False)
        environ = build_environ(self.scope, body)
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(self.sender, send)
            async with self.stream_send:
                await anyio.to_thread.run_sync(self.wsgi, environ, self.start_response)
        if self.exc_info is not None:
            raise self.exc_info[0].with_traceback(self.exc_info[1], self.exc_info[2])

    async def sender(self, send: Send) -> None:
        async with self.stream_receive:
            async for message in self.stream_receive:
                await send(message)

    def start_response(self, status: str, response_headers: typing.List[typing.Tuple[str, str]], exc_info: typing.Any=None) -> None:
        self.exc_info = exc_info
        if not self.response_started:
            self.response_started = True
            status_code_string, _ = status.split(' ', 1)
            status_code = int(status_code_string)
            headers = [(name.strip().encode('ascii').lower(), value.strip().encode('ascii')) for name, value in response_headers]
            anyio.from_thread.run(self.stream_send.send, {'type': 'http.response.start', 'status': status_code, 'headers': headers})

    def wsgi(self, environ: typing.Dict[str, typing.Any], start_response: typing.Callable[..., typing.Any]) -> None:
        for chunk in self.app(environ, start_response):
            anyio.from_thread.run(self.stream_send.send, {'type': 'http.response.body', 'body': chunk, 'more_body': True})
        anyio.from_thread.run(self.stream_send.send, {'type': 'http.response.body', 'body': b''})