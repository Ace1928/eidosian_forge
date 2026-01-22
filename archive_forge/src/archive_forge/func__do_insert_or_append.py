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
def _do_insert_or_append(self, event_key: _EventKey[_ET], is_append: bool) -> None:
    target = event_key.dispatch_target
    assert isinstance(target, type), 'Class-level Event targets must be classes.'
    if not getattr(target, '_sa_propagate_class_events', True):
        raise exc.InvalidRequestError(f"Can't assign an event directly to the {target} class")
    cls: Type[_ET]
    for cls in util.walk_subclasses(target):
        if cls is not target and cls not in self._clslevel:
            self.update_subclass(cls)
        else:
            if cls not in self._clslevel:
                self.update_subclass(cls)
            if is_append:
                self._clslevel[cls].append(event_key._listen_fn)
            else:
                self._clslevel[cls].appendleft(event_key._listen_fn)
    registry._stored_in_collection(event_key, self)