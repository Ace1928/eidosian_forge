from within the mutable extension::
from __future__ import annotations
from collections import defaultdict
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from weakref import WeakKeyDictionary
from .. import event
from .. import inspect
from .. import types
from .. import util
from ..orm import Mapper
from ..orm._typing import _ExternalEntityType
from ..orm._typing import _O
from ..orm._typing import _T
from ..orm.attributes import AttributeEventToken
from ..orm.attributes import flag_modified
from ..orm.attributes import InstrumentedAttribute
from ..orm.attributes import QueryableAttribute
from ..orm.context import QueryContext
from ..orm.decl_api import DeclarativeAttributeIntercept
from ..orm.state import InstanceState
from ..orm.unitofwork import UOWTransaction
from ..sql.base import SchemaEventTarget
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import memoized_property
from ..util.typing import SupportsIndex
from ..util.typing import TypeGuard
class MutableList(Mutable, List[_T]):
    """A list type that implements :class:`.Mutable`.

    The :class:`.MutableList` object implements a list that will
    emit change events to the underlying mapping when the contents of
    the list are altered, including when values are added or removed.

    Note that :class:`.MutableList` does **not** apply mutable tracking to  the
    *values themselves* inside the list. Therefore it is not a sufficient
    solution for the use case of tracking deep changes to a *recursive*
    mutable structure, such as a JSON structure.  To support this use case,
    build a subclass of  :class:`.MutableList` that provides appropriate
    coercion to the values placed in the dictionary so that they too are
    "mutable", and emit events up to their parent structure.

    .. seealso::

        :class:`.MutableDict`

        :class:`.MutableSet`

    """

    def __reduce_ex__(self, proto: SupportsIndex) -> Tuple[type, Tuple[List[int]]]:
        return (self.__class__, (list(self),))

    def __setstate__(self, state: Iterable[_T]) -> None:
        self[:] = state

    def is_scalar(self, value: _T | Iterable[_T]) -> TypeGuard[_T]:
        return not util.is_non_string_iterable(value)

    def is_iterable(self, value: _T | Iterable[_T]) -> TypeGuard[Iterable[_T]]:
        return util.is_non_string_iterable(value)

    def __setitem__(self, index: SupportsIndex | slice, value: _T | Iterable[_T]) -> None:
        """Detect list set events and emit change events."""
        if isinstance(index, SupportsIndex) and self.is_scalar(value):
            super().__setitem__(index, value)
        elif isinstance(index, slice) and self.is_iterable(value):
            super().__setitem__(index, value)
        self.changed()

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        """Detect list del events and emit change events."""
        super().__delitem__(index)
        self.changed()

    def pop(self, *arg: SupportsIndex) -> _T:
        result = super().pop(*arg)
        self.changed()
        return result

    def append(self, x: _T) -> None:
        super().append(x)
        self.changed()

    def extend(self, x: Iterable[_T]) -> None:
        super().extend(x)
        self.changed()

    def __iadd__(self, x: Iterable[_T]) -> MutableList[_T]:
        self.extend(x)
        return self

    def insert(self, i: SupportsIndex, x: _T) -> None:
        super().insert(i, x)
        self.changed()

    def remove(self, i: _T) -> None:
        super().remove(i)
        self.changed()

    def clear(self) -> None:
        super().clear()
        self.changed()

    def sort(self, **kw: Any) -> None:
        super().sort(**kw)
        self.changed()

    def reverse(self) -> None:
        super().reverse()
        self.changed()

    @classmethod
    def coerce(cls, key: str, value: MutableList[_T] | _T) -> Optional[MutableList[_T]]:
        """Convert plain list to instance of this class."""
        if not isinstance(value, cls):
            if isinstance(value, list):
                return cls(value)
            return Mutable.coerce(key, value)
        else:
            return value