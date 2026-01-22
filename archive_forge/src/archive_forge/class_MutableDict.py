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
class MutableDict(Mutable, Dict[_KT, _VT]):
    """A dictionary type that implements :class:`.Mutable`.

    The :class:`.MutableDict` object implements a dictionary that will
    emit change events to the underlying mapping when the contents of
    the dictionary are altered, including when values are added or removed.

    Note that :class:`.MutableDict` does **not** apply mutable tracking to  the
    *values themselves* inside the dictionary. Therefore it is not a sufficient
    solution for the use case of tracking deep changes to a *recursive*
    dictionary structure, such as a JSON structure.  To support this use case,
    build a subclass of  :class:`.MutableDict` that provides appropriate
    coercion to the values placed in the dictionary so that they too are
    "mutable", and emit events up to their parent structure.

    .. seealso::

        :class:`.MutableList`

        :class:`.MutableSet`

    """

    def __setitem__(self, key: _KT, value: _VT) -> None:
        """Detect dictionary set events and emit change events."""
        super().__setitem__(key, value)
        self.changed()
    if TYPE_CHECKING:

        @overload
        def setdefault(self: MutableDict[_KT, Optional[_T]], key: _KT, value: None=None) -> Optional[_T]:
            ...

        @overload
        def setdefault(self, key: _KT, value: _VT) -> _VT:
            ...

        def setdefault(self, key: _KT, value: object=None) -> object:
            ...
    else:

        def setdefault(self, *arg):
            result = super().setdefault(*arg)
            self.changed()
            return result

    def __delitem__(self, key: _KT) -> None:
        """Detect dictionary del events and emit change events."""
        super().__delitem__(key)
        self.changed()

    def update(self, *a: Any, **kw: _VT) -> None:
        super().update(*a, **kw)
        self.changed()
    if TYPE_CHECKING:

        @overload
        def pop(self, __key: _KT) -> _VT:
            ...

        @overload
        def pop(self, __key: _KT, __default: _VT | _T) -> _VT | _T:
            ...

        def pop(self, __key: _KT, __default: _VT | _T | None=None) -> _VT | _T:
            ...
    else:

        def pop(self, *arg):
            result = super().pop(*arg)
            self.changed()
            return result

    def popitem(self) -> Tuple[_KT, _VT]:
        result = super().popitem()
        self.changed()
        return result

    def clear(self) -> None:
        super().clear()
        self.changed()

    @classmethod
    def coerce(cls, key: str, value: Any) -> MutableDict[_KT, _VT] | None:
        """Convert plain dictionary to instance of this class."""
        if not isinstance(value, cls):
            if isinstance(value, dict):
                return cls(value)
            return Mutable.coerce(key, value)
        else:
            return value

    def __getstate__(self) -> Dict[_KT, _VT]:
        return dict(self)

    def __setstate__(self, state: Union[Dict[str, int], Dict[str, str]]) -> None:
        self.update(state)