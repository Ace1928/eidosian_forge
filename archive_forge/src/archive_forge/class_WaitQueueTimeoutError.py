from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class WaitQueueTimeoutError(ConnectionFailure):
    """Raised when an operation times out waiting to checkout a connection from the pool.

    Subclass of :exc:`~pymongo.errors.ConnectionFailure`.

    .. versionadded:: 4.2
    """

    @property
    def timeout(self) -> bool:
        return True