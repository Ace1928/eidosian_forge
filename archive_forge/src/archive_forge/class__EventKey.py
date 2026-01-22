from __future__ import annotations
import collections
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import weakref
from .. import exc
from .. import util
class _EventKey(Generic[_ET]):
    """Represent :func:`.listen` arguments."""
    __slots__ = ('target', 'identifier', 'fn', 'fn_key', 'fn_wrap', 'dispatch_target')
    target: _ET
    identifier: str
    fn: _ListenerFnType
    fn_key: _ListenerFnKeyType
    dispatch_target: Any
    _fn_wrap: Optional[_ListenerFnType]

    def __init__(self, target: _ET, identifier: str, fn: _ListenerFnType, dispatch_target: Any, _fn_wrap: Optional[_ListenerFnType]=None):
        self.target = target
        self.identifier = identifier
        self.fn = fn
        if isinstance(fn, types.MethodType):
            self.fn_key = (id(fn.__func__), id(fn.__self__))
        else:
            self.fn_key = id(fn)
        self.fn_wrap = _fn_wrap
        self.dispatch_target = dispatch_target

    @property
    def _key(self) -> _EventKeyTupleType:
        return (id(self.target), self.identifier, self.fn_key)

    def with_wrapper(self, fn_wrap: _ListenerFnType) -> _EventKey[_ET]:
        if fn_wrap is self._listen_fn:
            return self
        else:
            return _EventKey(self.target, self.identifier, self.fn, self.dispatch_target, _fn_wrap=fn_wrap)

    def with_dispatch_target(self, dispatch_target: Any) -> _EventKey[_ET]:
        if dispatch_target is self.dispatch_target:
            return self
        else:
            return _EventKey(self.target, self.identifier, self.fn, dispatch_target, _fn_wrap=self.fn_wrap)

    def listen(self, *args: Any, **kw: Any) -> None:
        once = kw.pop('once', False)
        once_unless_exception = kw.pop('_once_unless_exception', False)
        named = kw.pop('named', False)
        target, identifier, fn = (self.dispatch_target, self.identifier, self._listen_fn)
        dispatch_collection = getattr(target.dispatch, identifier)
        adjusted_fn = dispatch_collection._adjust_fn_spec(fn, named)
        self = self.with_wrapper(adjusted_fn)
        stub_function = getattr(self.dispatch_target.dispatch._events, self.identifier)
        if hasattr(stub_function, '_sa_warn'):
            stub_function._sa_warn()
        if once or once_unless_exception:
            self.with_wrapper(util.only_once(self._listen_fn, retry_on_exception=once_unless_exception)).listen(*args, **kw)
        else:
            self.dispatch_target.dispatch._listen(self, *args, **kw)

    def remove(self) -> None:
        key = self._key
        if key not in _key_to_collection:
            raise exc.InvalidRequestError('No listeners found for event %s / %r / %s ' % (self.target, self.identifier, self.fn))
        dispatch_reg = _key_to_collection.pop(key)
        for collection_ref, listener_ref in dispatch_reg.items():
            collection = collection_ref()
            listener_fn = listener_ref()
            if collection is not None and listener_fn is not None:
                collection.remove(self.with_wrapper(listener_fn))

    def contains(self) -> bool:
        """Return True if this event key is registered to listen."""
        return self._key in _key_to_collection

    def base_listen(self, propagate: bool=False, insert: bool=False, named: bool=False, retval: Optional[bool]=None, asyncio: bool=False) -> None:
        target, identifier = (self.dispatch_target, self.identifier)
        dispatch_collection = getattr(target.dispatch, identifier)
        for_modify = dispatch_collection.for_modify(target.dispatch)
        if asyncio:
            for_modify._set_asyncio()
        if insert:
            for_modify.insert(self, propagate)
        else:
            for_modify.append(self, propagate)

    @property
    def _listen_fn(self) -> _ListenerFnType:
        return self.fn_wrap or self.fn

    def append_to_list(self, owner: RefCollection[_ET], list_: Deque[_ListenerFnType]) -> bool:
        if _stored_in_collection(self, owner):
            list_.append(self._listen_fn)
            return True
        else:
            return False

    def remove_from_list(self, owner: RefCollection[_ET], list_: Deque[_ListenerFnType]) -> None:
        _removed_from_collection(self, owner)
        list_.remove(self._listen_fn)

    def prepend_to_list(self, owner: RefCollection[_ET], list_: Deque[_ListenerFnType]) -> bool:
        if _stored_in_collection(self, owner):
            list_.appendleft(self._listen_fn)
            return True
        else:
            return False