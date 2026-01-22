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
class _EmptyListener(_InstanceLevelDispatch[_ET]):
    """Serves as a proxy interface to the events
    served by a _ClsLevelDispatch, when there are no
    instance-level events present.

    Is replaced by _ListenerCollection when instance-level
    events are added.

    """
    __slots__ = ('parent', 'parent_listeners', 'name')
    propagate: FrozenSet[_ListenerFnType] = frozenset()
    listeners: Tuple[()] = ()
    parent: _ClsLevelDispatch[_ET]
    parent_listeners: _ListenerFnSequenceType[_ListenerFnType]
    name: str

    def __init__(self, parent: _ClsLevelDispatch[_ET], target_cls: Type[_ET]):
        if target_cls not in parent._clslevel:
            parent.update_subclass(target_cls)
        self.parent = parent
        self.parent_listeners = parent._clslevel[target_cls]
        self.name = parent.name

    def for_modify(self, obj: _DispatchCommon[_ET]) -> _ListenerCollection[_ET]:
        """Return an event collection which can be modified.

        For _EmptyListener at the instance level of
        a dispatcher, this generates a new
        _ListenerCollection, applies it to the instance,
        and returns it.

        """
        obj = cast('_Dispatch[_ET]', obj)
        assert obj._instance_cls is not None
        result = _ListenerCollection(self.parent, obj._instance_cls)
        if getattr(obj, self.name) is self:
            setattr(obj, self.name, result)
        else:
            assert isinstance(getattr(obj, self.name), _JoinedListener)
        return result

    def _needs_modify(self, *args: Any, **kw: Any) -> NoReturn:
        raise NotImplementedError('need to call for_modify()')

    def exec_once(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def exec_once_unless_exception(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def insert(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def append(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def remove(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def clear(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def __call__(self, *args: Any, **kw: Any) -> None:
        """Execute this event."""
        for fn in self.parent_listeners:
            fn(*args, **kw)

    def __contains__(self, item: Any) -> bool:
        return item in self.parent_listeners

    def __len__(self) -> int:
        return len(self.parent_listeners)

    def __iter__(self) -> Iterator[_ListenerFnType]:
        return iter(self.parent_listeners)

    def __bool__(self) -> bool:
        return bool(self.parent_listeners)