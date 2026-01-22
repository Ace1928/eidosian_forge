from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union
from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource
from ._tasks import TaskGroup
class UnreliableObjectSendStream(Generic[T_contra], AsyncResource, TypedAttributeProvider):
    """
    An interface for sending objects.

    This interface makes no guarantees that the messages sent will reach the
    recipient(s) in the same order in which they were sent, or at all.
    """

    @abstractmethod
    async def send(self, item: T_contra) -> None:
        """
        Send an item to the peer(s).

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if the send stream has been explicitly
            closed
        :raises ~anyio.BrokenResourceError: if this stream has been rendered unusable
            due to external causes
        """