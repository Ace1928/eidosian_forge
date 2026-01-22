from __future__ import annotations
import collections
from itertools import chain
import threading
from types import TracebackType
import typing
from typing import Any
from typing import cast
from typing import Collection
from typing import Deque
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import MutableMapping
from typing import MutableSequence
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import legacy
from . import registry
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
from ..util.concurrency import AsyncAdaptedLock
from ..util.typing import Protocol
class _ListenerCollection(_CompoundListener[_ET]):
    """Instance-level attributes on instances of :class:`._Dispatch`.

    Represents a collection of listeners.

    As of 0.7.9, _ListenerCollection is only first
    created via the _EmptyListener.for_modify() method.

    """
    __slots__ = ('parent_listeners', 'parent', 'name', 'listeners', 'propagate', '__weakref__')
    parent_listeners: Collection[_ListenerFnType]
    parent: _ClsLevelDispatch[_ET]
    name: str
    listeners: Deque[_ListenerFnType]
    propagate: Set[_ListenerFnType]

    def __init__(self, parent: _ClsLevelDispatch[_ET], target_cls: Type[_ET]):
        super().__init__()
        if target_cls not in parent._clslevel:
            parent.update_subclass(target_cls)
        self._exec_once = False
        self._exec_w_sync_once = False
        self.parent_listeners = parent._clslevel[target_cls]
        self.parent = parent
        self.name = parent.name
        self.listeners = collections.deque()
        self.propagate = set()

    def for_modify(self, obj: _DispatchCommon[_ET]) -> _ListenerCollection[_ET]:
        """Return an event collection which can be modified.

        For _ListenerCollection at the instance level of
        a dispatcher, this returns self.

        """
        return self

    def _update(self, other: _ListenerCollection[_ET], only_propagate: bool=True) -> None:
        """Populate from the listeners in another :class:`_Dispatch`
        object."""
        existing_listeners = self.listeners
        existing_listener_set = set(existing_listeners)
        self.propagate.update(other.propagate)
        other_listeners = [l for l in other.listeners if l not in existing_listener_set and (not only_propagate) or l in self.propagate]
        existing_listeners.extend(other_listeners)
        if other._is_asyncio:
            self._set_asyncio()
        to_associate = other.propagate.union(other_listeners)
        registry._stored_in_collection_multi(self, other, to_associate)

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        if event_key.prepend_to_list(self, self.listeners):
            if propagate:
                self.propagate.add(event_key._listen_fn)

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        if event_key.append_to_list(self, self.listeners):
            if propagate:
                self.propagate.add(event_key._listen_fn)

    def remove(self, event_key: _EventKey[_ET]) -> None:
        self.listeners.remove(event_key._listen_fn)
        self.propagate.discard(event_key._listen_fn)
        registry._removed_from_collection(event_key, self)

    def clear(self) -> None:
        registry._clear(self, self.listeners)
        self.propagate.clear()
        self.listeners.clear()