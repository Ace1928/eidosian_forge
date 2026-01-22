from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class _ServerEvent:
    """Base class for server events."""
    __slots__ = ('__server_address', '__topology_id')

    def __init__(self, server_address: _Address, topology_id: ObjectId) -> None:
        self.__server_address = server_address
        self.__topology_id = topology_id

    @property
    def server_address(self) -> _Address:
        """The address (host, port) pair of the server"""
        return self.__server_address

    @property
    def topology_id(self) -> ObjectId:
        """A unique identifier for the topology this server is a part of."""
        return self.__topology_id

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.server_address} topology_id: {self.topology_id}>'