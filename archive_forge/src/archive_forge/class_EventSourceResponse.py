import io
import logging
import re
from datetime import datetime, timezone
from functools import partial
from typing import (
import anyio
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
class EventSourceResponse(Response):
    """Implements the ServerSentEvent Protocol:
    https://www.w3.org/TR/2009/WD-eventsource-20090421/

    Responses must not be compressed by middleware in order to work.
    implementation based on Starlette StreamingResponse
    """
    body_iterator: AsyncContentStream
    DEFAULT_PING_INTERVAL = 15

    def __init__(self, content: ContentStream, status_code: int=200, headers: Optional[Mapping[str, str]]=None, media_type: str='text/event-stream', background: Optional[BackgroundTask]=None, ping: Optional[int]=None, sep: Optional[str]=None, ping_message_factory: Optional[Callable[[], ServerSentEvent]]=None, data_sender_callable: Optional[Callable[[], Coroutine[None, None, None]]]=None, send_timeout: Optional[float]=None) -> None:
        if sep is not None and sep not in ['\r\n', '\r', '\n']:
            raise ValueError(f'sep must be one of: \\r\\n, \\r, \\n, got: {sep}')
        self.DEFAULT_SEPARATOR = '\r\n'
        self.sep = sep if sep is not None else self.DEFAULT_SEPARATOR
        self.ping_message_factory = ping_message_factory
        if isinstance(content, AsyncIterable):
            self.body_iterator = content
        else:
            self.body_iterator = iterate_in_threadpool(content)
        self.status_code = status_code
        self.media_type = self.media_type if media_type is None else media_type
        self.background = background
        self.data_sender_callable = data_sender_callable
        self.send_timeout = send_timeout
        _headers: dict[str, str] = {}
        if headers is not None:
            _headers.update(headers)
        _headers.setdefault('Cache-Control', 'no-cache')
        _headers['Connection'] = 'keep-alive'
        _headers['X-Accel-Buffering'] = 'no'
        self.init_headers(_headers)
        self.ping_interval = self.DEFAULT_PING_INTERVAL if ping is None else ping
        self.active = True
        self._ping_task = None
        self._send_lock = anyio.Lock()

    @staticmethod
    async def listen_for_disconnect(receive: Receive) -> None:
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                _log.debug('Got event: http.disconnect. Stop streaming.')
                break

    @staticmethod
    async def listen_for_exit_signal() -> None:
        if AppStatus.should_exit:
            return
        if AppStatus.should_exit_event is None:
            AppStatus.should_exit_event = anyio.Event()
        if AppStatus.should_exit:
            return
        await AppStatus.should_exit_event.wait()

    async def stream_response(self, send: Send) -> None:
        await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
        async for data in self.body_iterator:
            chunk = ensure_bytes(data, self.sep)
            _log.debug('chunk: %s', chunk)
            with anyio.move_on_after(self.send_timeout) as timeout:
                await send({'type': 'http.response.body', 'body': chunk, 'more_body': True})
            if timeout.cancel_called:
                if hasattr(self.body_iterator, 'aclose'):
                    await self.body_iterator.aclose()
                raise SendTimeoutError()
        async with self._send_lock:
            self.active = False
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with anyio.create_task_group() as task_group:

            async def wrap(func: Callable[[], Awaitable[None]]) -> None:
                await func()
                task_group.cancel_scope.cancel()
            task_group.start_soon(wrap, partial(self.stream_response, send))
            task_group.start_soon(wrap, partial(self._ping, send))
            task_group.start_soon(wrap, self.listen_for_exit_signal)
            if self.data_sender_callable:
                task_group.start_soon(self.data_sender_callable)
            await wrap(partial(self.listen_for_disconnect, receive))
        if self.background is not None:
            await self.background()

    def enable_compression(self, force: bool=False) -> None:
        raise NotImplementedError

    @property
    def ping_interval(self) -> Union[int, float]:
        """Time interval between two ping massages"""
        return self._ping_interval

    @ping_interval.setter
    def ping_interval(self, value: Union[int, float]) -> None:
        """Setter for ping_interval property.

        :param int value: interval in sec between two ping values.
        """
        if not isinstance(value, (int, float)):
            raise TypeError('ping interval must be int')
        if value < 0:
            raise ValueError('ping interval must be greater then 0')
        self._ping_interval = value

    async def _ping(self, send: Send) -> None:
        while self.active:
            await anyio.sleep(self._ping_interval)
            if self.ping_message_factory:
                assert isinstance(self.ping_message_factory, Callable)
            ping = ServerSentEvent(comment=f'ping - {datetime.now(timezone.utc)}', sep=self.sep).encode() if self.ping_message_factory is None else ensure_bytes(self.ping_message_factory(), self.sep)
            _log.debug('ping: %s', ping)
            async with self._send_lock:
                if self.active:
                    await send({'type': 'http.response.body', 'body': ping, 'more_body': True})