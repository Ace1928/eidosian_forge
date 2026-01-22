from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class PoolCreatedEvent(_PoolEvent):
    """Published when a Connection Pool is created.

    :Parameters:
     - `address`: The address (host, port) pair of the server this Pool is
       attempting to connect to.

    .. versionadded:: 3.9
    """
    __slots__ = ('__options',)

    def __init__(self, address: _Address, options: dict[str, Any]) -> None:
        super().__init__(address)
        self.__options = options

    @property
    def options(self) -> dict[str, Any]:
        """Any non-default pool options that were set on this Connection Pool."""
        return self.__options

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.address!r}, {self.__options!r})'