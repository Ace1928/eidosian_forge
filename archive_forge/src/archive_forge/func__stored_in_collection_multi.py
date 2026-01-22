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
def _stored_in_collection_multi(newowner: RefCollection[_ET], oldowner: RefCollection[_ET], elements: Iterable[_ListenerFnType]) -> None:
    if not elements:
        return
    oldowner_ref = oldowner.ref
    newowner_ref = newowner.ref
    old_listener_to_key = _collection_to_key[oldowner_ref]
    new_listener_to_key = _collection_to_key[newowner_ref]
    for listen_fn in elements:
        listen_ref = weakref.ref(listen_fn)
        try:
            key = old_listener_to_key[listen_ref]
        except KeyError:
            continue
        try:
            dispatch_reg = _key_to_collection[key]
        except KeyError:
            continue
        if newowner_ref in dispatch_reg:
            assert dispatch_reg[newowner_ref] == listen_ref
        else:
            dispatch_reg[newowner_ref] = listen_ref
        new_listener_to_key[listen_ref] = key