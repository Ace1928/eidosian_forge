from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import base
from . import exc as orm_exc
from . import interfaces
from ._typing import _O
from ._typing import is_collection_impl
from .base import ATTR_WAS_SET
from .base import INIT_OK
from .base import LoaderCallableStatus
from .base import NEVER_SET
from .base import NO_VALUE
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import SQL_OK
from .path_registry import PathRegistry
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class AttributeState:
    """Provide an inspection interface corresponding
    to a particular attribute on a particular mapped object.

    The :class:`.AttributeState` object is accessed
    via the :attr:`.InstanceState.attrs` collection
    of a particular :class:`.InstanceState`::

        from sqlalchemy import inspect

        insp = inspect(some_mapped_object)
        attr_state = insp.attrs.some_attribute

    """
    __slots__ = ('state', 'key')
    state: InstanceState[Any]
    key: str

    def __init__(self, state: InstanceState[Any], key: str):
        self.state = state
        self.key = key

    @property
    def loaded_value(self) -> Any:
        """The current value of this attribute as loaded from the database.

        If the value has not been loaded, or is otherwise not present
        in the object's dictionary, returns NO_VALUE.

        """
        return self.state.dict.get(self.key, NO_VALUE)

    @property
    def value(self) -> Any:
        """Return the value of this attribute.

        This operation is equivalent to accessing the object's
        attribute directly or via ``getattr()``, and will fire
        off any pending loader callables if needed.

        """
        return self.state.manager[self.key].__get__(self.state.obj(), self.state.class_)

    @property
    def history(self) -> History:
        """Return the current **pre-flush** change history for
        this attribute, via the :class:`.History` interface.

        This method will **not** emit loader callables if the value of the
        attribute is unloaded.

        .. note::

            The attribute history system tracks changes on a **per flush
            basis**. Each time the :class:`.Session` is flushed, the history
            of each attribute is reset to empty.   The :class:`.Session` by
            default autoflushes each time a :class:`_query.Query` is invoked.
            For
            options on how to control this, see :ref:`session_flushing`.


        .. seealso::

            :meth:`.AttributeState.load_history` - retrieve history
            using loader callables if the value is not locally present.

            :func:`.attributes.get_history` - underlying function

        """
        return self.state.get_history(self.key, PASSIVE_NO_INITIALIZE)

    def load_history(self) -> History:
        """Return the current **pre-flush** change history for
        this attribute, via the :class:`.History` interface.

        This method **will** emit loader callables if the value of the
        attribute is unloaded.

        .. note::

            The attribute history system tracks changes on a **per flush
            basis**. Each time the :class:`.Session` is flushed, the history
            of each attribute is reset to empty.   The :class:`.Session` by
            default autoflushes each time a :class:`_query.Query` is invoked.
            For
            options on how to control this, see :ref:`session_flushing`.

        .. seealso::

            :attr:`.AttributeState.history`

            :func:`.attributes.get_history` - underlying function

        """
        return self.state.get_history(self.key, PASSIVE_OFF ^ INIT_OK)