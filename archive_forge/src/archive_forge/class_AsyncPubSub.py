import logging
import asyncio
import copy
import inspect
import re
import warnings
from typing import (
from aiokeydb.v1.asyncio.connection import (
from aiokeydb.v1.asyncio.lock import AsyncLock
from aiokeydb.v1.asyncio.retry import Retry
from aiokeydb.v1.core import (
from aiokeydb.v1.commands import (
from aiokeydb.v1.compat import Protocol, TypedDict
from aiokeydb.v1.credentials import CredentialProvider
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.typing import ChannelT, EncodableT, KeyT
from aiokeydb.v1.utils import safe_str, str_if_bytes
class AsyncPubSub:
    """
    AsyncPubSub provides publish, subscribe and listen support to KeyDB channels.

    After subscribing to one or more channels, the listen() method will block
    until a message arrives on one of the subscribed channels. That message
    will be returned and it's safe to start listening again.
    """
    PUBLISH_MESSAGE_TYPES = ('message', 'pmessage')
    UNSUBSCRIBE_MESSAGE_TYPES = ('unsubscribe', 'punsubscribe')
    HEALTH_CHECK_MESSAGE = 'aiokeydb-py-health-check'

    def __init__(self, connection_pool: AsyncConnectionPool, shard_hint: Optional[str]=None, ignore_subscribe_messages: bool=False, encoder=None):
        self.connection_pool = connection_pool
        self.shard_hint = shard_hint
        self.ignore_subscribe_messages = ignore_subscribe_messages
        self.connection = None
        self.encoder = encoder
        if self.encoder is None:
            self.encoder = self.connection_pool.get_encoder()
        if self.encoder.decode_responses:
            self.health_check_response: Iterable[Union[str, bytes]] = ['pong', self.HEALTH_CHECK_MESSAGE]
        else:
            self.health_check_response = [b'pong', self.encoder.encode(self.HEALTH_CHECK_MESSAGE)]
        self.channels = {}
        self.pending_unsubscribe_channels = set()
        self.patterns = {}
        self.pending_unsubscribe_patterns = set()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.reset()

    def __del__(self):
        if self.connection:
            self.connection.clear_connect_callbacks()

    async def reset(self):
        async with self._lock:
            if self.connection:
                await self.connection.disconnect()
                self.connection.clear_connect_callbacks()
                await self.connection_pool.release(self.connection)
                self.connection = None
            self.channels = {}
            self.pending_unsubscribe_channels = set()
            self.patterns = {}
            self.pending_unsubscribe_patterns = set()

    def close(self) -> Awaitable[NoReturn]:
        return self.reset()

    async def on_connect(self, connection: AsyncConnection):
        """Re-subscribe to any channels and patterns previously subscribed to"""
        self.pending_unsubscribe_channels.clear()
        self.pending_unsubscribe_patterns.clear()
        if self.channels:
            channels = {}
            for k, v in self.channels.items():
                channels[self.encoder.decode(k, force=True)] = v
            await self.subscribe(**channels)
        if self.patterns:
            patterns = {}
            for k, v in self.patterns.items():
                patterns[self.encoder.decode(k, force=True)] = v
            await self.psubscribe(**patterns)

    @property
    def subscribed(self):
        """Indicates if there are subscriptions to any channels or patterns"""
        return bool(self.channels or self.patterns)

    async def execute_command(self, *args: EncodableT):
        """Execute a publish/subscribe command"""
        await self.connect()
        connection = self.connection
        kwargs = {'check_health': not self.subscribed}
        await self._execute(connection, connection.send_command, *args, **kwargs)

    async def connect(self):
        """
        Ensure that the PubSub is connected
        """
        if self.connection is None:
            self.connection = await self.connection_pool.get_connection('pubsub', self.shard_hint)
            self.connection.register_connect_callback(self.on_connect)
        else:
            await self.connection.connect()

    async def _disconnect_raise_connect(self, conn, error):
        """
        Close the connection and raise an exception
        if retry_on_timeout is not set or the error
        is not a TimeoutError. Otherwise, try to reconnect
        """
        await conn.disconnect()
        if not (conn.retry_on_timeout and isinstance(error, TimeoutError)):
            raise error
        await conn.connect()

    async def _execute(self, conn, command, *args, **kwargs):
        """
        Connect manually upon disconnection. If the KeyDB server is down,
        this will fail and raise a ConnectionError as desired.
        After reconnection, the ``on_connect`` callback should have been
        called by the # connection to resubscribe us to any channels and
        patterns we were previously listening to
        """
        return await conn.retry.call_with_retry(lambda: command(*args, **kwargs), lambda error: self._disconnect_raise_connect(conn, error))

    async def parse_response(self, block: bool=True, timeout: float=0):
        """Parse the response from a publish/subscribe command"""
        conn = self.connection
        if conn is None:
            raise RuntimeError('pubsub connection not set: did you forget to call subscribe() or psubscribe()?')
        await self.check_health()

        async def try_read():
            if not block:
                if not await conn.can_read(timeout=timeout):
                    return None
            else:
                await conn.connect()
            return await conn.read_response()
        response = await self._execute(conn, try_read)
        if conn.health_check_interval and response == self.health_check_response:
            return None
        return response

    async def check_health(self):
        conn = self.connection
        if conn is None:
            raise RuntimeError('pubsub connection not set: did you forget to call subscribe() or psubscribe()?')
        if conn.health_check_interval and asyncio.get_event_loop().time() > conn.next_health_check:
            await conn.send_command('PING', self.HEALTH_CHECK_MESSAGE, check_health=False)

    def _normalize_keys(self, data: _NormalizeKeysT) -> _NormalizeKeysT:
        """
        normalize channel/pattern names to be either bytes or strings
        based on whether responses are automatically decoded. this saves us
        from coercing the value for each message coming in.
        """
        encode = self.encoder.encode
        decode = self.encoder.decode
        return {decode(encode(k)): v for k, v in data.items()}

    async def psubscribe(self, *args: ChannelT, **kwargs: PubSubHandler):
        """
        Subscribe to channel patterns. Patterns supplied as keyword arguments
        expect a pattern name as the key and a callable as the value. A
        pattern's callable will be invoked automatically when a message is
        received on that pattern rather than producing a message via
        ``listen()``.
        """
        parsed_args = list_or_args((args[0],), args[1:]) if args else args
        new_patterns: Dict[ChannelT, PubSubHandler] = dict.fromkeys(parsed_args)
        new_patterns.update(kwargs)
        ret_val = await self.execute_command('PSUBSCRIBE', *new_patterns.keys())
        new_patterns = self._normalize_keys(new_patterns)
        self.patterns.update(new_patterns)
        self.pending_unsubscribe_patterns.difference_update(new_patterns)
        return ret_val

    def punsubscribe(self, *args: ChannelT) -> Awaitable:
        """
        Unsubscribe from the supplied patterns. If empty, unsubscribe from
        all patterns.
        """
        patterns: Iterable[ChannelT]
        if args:
            parsed_args = list_or_args((args[0],), args[1:])
            patterns = self._normalize_keys(dict.fromkeys(parsed_args)).keys()
        else:
            parsed_args = []
            patterns = self.patterns
        self.pending_unsubscribe_patterns.update(patterns)
        return self.execute_command('PUNSUBSCRIBE', *parsed_args)

    async def subscribe(self, *args: ChannelT, **kwargs: Callable):
        """
        Subscribe to channels. Channels supplied as keyword arguments expect
        a channel name as the key and a callable as the value. A channel's
        callable will be invoked automatically when a message is received on
        that channel rather than producing a message via ``listen()`` or
        ``get_message()``.
        """
        parsed_args = list_or_args((args[0],), args[1:]) if args else ()
        new_channels = dict.fromkeys(parsed_args)
        new_channels.update(kwargs)
        ret_val = await self.execute_command('SUBSCRIBE', *new_channels.keys())
        new_channels = self._normalize_keys(new_channels)
        self.channels.update(new_channels)
        self.pending_unsubscribe_channels.difference_update(new_channels)
        return ret_val

    def unsubscribe(self, *args) -> Awaitable:
        """
        Unsubscribe from the supplied channels. If empty, unsubscribe from
        all channels
        """
        if args:
            parsed_args = list_or_args(args[0], args[1:])
            channels = self._normalize_keys(dict.fromkeys(parsed_args))
        else:
            parsed_args = []
            channels = self.channels
        self.pending_unsubscribe_channels.update(channels)
        return self.execute_command('UNSUBSCRIBE', *parsed_args)

    async def listen(self) -> AsyncIterator:
        """Listen for messages on channels this client has been subscribed to"""
        while self.subscribed:
            response = await self.handle_message(await self.parse_response(block=True))
            if response is not None:
                yield response

    async def get_message(self, ignore_subscribe_messages: bool=False, timeout: float=0.0):
        """
        Get the next message if one is available, otherwise None.

        If timeout is specified, the system will wait for `timeout` seconds
        before returning. Timeout should be specified as a floating point
        number.
        """
        response = await self.parse_response(block=False, timeout=timeout)
        if response:
            return await self.handle_message(response, ignore_subscribe_messages)
        return None

    def ping(self, message=None) -> Awaitable:
        """
        Ping the KeyDB server
        """
        message = '' if message is None else message
        return self.execute_command('PING', message)

    async def handle_message(self, response, ignore_subscribe_messages=False):
        """
        Parses a pub/sub message. If the channel or pattern was subscribed to
        with a message handler, the handler is invoked instead of a parsed
        message being returned.
        """
        message_type = str_if_bytes(response[0])
        if message_type == 'pmessage':
            message = {'type': message_type, 'pattern': response[1], 'channel': response[2], 'data': response[3]}
        elif message_type == 'pong':
            message = {'type': message_type, 'pattern': None, 'channel': None, 'data': response[1]}
        else:
            message = {'type': message_type, 'pattern': None, 'channel': response[1], 'data': response[2]}
        if message_type in self.UNSUBSCRIBE_MESSAGE_TYPES:
            if message_type == 'punsubscribe':
                pattern = response[1]
                if pattern in self.pending_unsubscribe_patterns:
                    self.pending_unsubscribe_patterns.remove(pattern)
                    self.patterns.pop(pattern, None)
            else:
                channel = response[1]
                if channel in self.pending_unsubscribe_channels:
                    self.pending_unsubscribe_channels.remove(channel)
                    self.channels.pop(channel, None)
        if message_type in self.PUBLISH_MESSAGE_TYPES:
            if message_type == 'pmessage':
                handler = self.patterns.get(message['pattern'], None)
            else:
                handler = self.channels.get(message['channel'], None)
            if handler:
                if inspect.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
                return None
        elif message_type != 'pong':
            if ignore_subscribe_messages or self.ignore_subscribe_messages:
                return None
        return message

    async def run(self, *, exception_handler: Optional['PSWorkerThreadExcHandlerT']=None, poll_timeout: float=1.0) -> None:
        """Process pub/sub messages using registered callbacks.

        This is the equivalent of :py:meth:`redis.PubSub.run_in_thread` in
        redis-py, but it is a coroutine. To launch it as a separate task, use
        ``asyncio.create_task``:

            >>> task = asyncio.create_task(pubsub.run())

        To shut it down, use asyncio cancellation:

            >>> task.cancel()
            >>> await task
        """
        for channel, handler in self.channels.items():
            if handler is None:
                raise PubSubError(f"Channel: '{channel}' has no handler registered")
        for pattern, handler in self.patterns.items():
            if handler is None:
                raise PubSubError(f"Pattern: '{pattern}' has no handler registered")
        await self.connect()
        while True:
            try:
                await self.get_message(ignore_subscribe_messages=True, timeout=poll_timeout)
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                if exception_handler is None:
                    raise
                res = exception_handler(e, self)
                if inspect.isawaitable(res):
                    await res
            await asyncio.sleep(0)