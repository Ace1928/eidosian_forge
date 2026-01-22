from __future__ import annotations
import typing
import logging
import datetime
from redis.commands import (
from redis.client import (
from redis.asyncio.client import (
from aiokeydb.v2.connection import (
from typing import Any, Iterable, Mapping, Callable, Union, overload, TYPE_CHECKING
from typing_extensions import Literal
from .utils.helpers import get_retryable_wrapper
class RetryablePubSub(PubSub):
    """
    Retryable PubSub
    """

    @retryable_wrapper
    def subscribe(self, *args: 'ChannelT', **kwargs: typing.Callable):
        """
        Subscribe to channels. Channels supplied as keyword arguments expect
        a channel name as the key and a callable as the value. A channel's
        callable will be invoked automatically when a message is received on
        that channel rather than producing a message via ``listen()`` or
        ``get_message()``.
        """
        return super().subscribe(*args, **kwargs)

    @retryable_wrapper
    def unsubscribe(self, *args):
        """
        Unsubscribe from the supplied channels. If empty, unsubscribe from
        all channels
        """
        return super().unsubscribe(*args)

    @retryable_wrapper
    def listen(self) -> typing.Iterator:
        """Listen for messages on channels this client has been subscribed to"""
        yield from super().listen()