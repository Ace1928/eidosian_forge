from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union
from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource
from ._tasks import TaskGroup
class ObjectStream(ObjectReceiveStream[T_Item], ObjectSendStream[T_Item], UnreliableObjectStream[T_Item]):
    """
    A bidirectional message stream which guarantees the order and reliability of message
    delivery.
    """

    @abstractmethod
    async def send_eof(self) -> None:
        """
        Send an end-of-file indication to the peer.

        You should not try to send any further data to this stream after calling this
        method. This method is idempotent (does nothing on successive calls).
        """