from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class TopologyEvent:
    """Base class for topology description events."""
    __slots__ = ('__topology_id',)

    def __init__(self, topology_id: ObjectId) -> None:
        self.__topology_id = topology_id

    @property
    def topology_id(self) -> ObjectId:
        """A unique identifier for the topology this server is a part of."""
        return self.__topology_id

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} topology_id: {self.topology_id}>'