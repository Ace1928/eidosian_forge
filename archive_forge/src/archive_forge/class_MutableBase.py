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
class MutableBase:
    """Common base class to :class:`.Mutable`
    and :class:`.MutableComposite`.

    """

    @memoized_property
    def _parents(self) -> WeakKeyDictionary[Any, Any]:
        """Dictionary of parent object's :class:`.InstanceState`->attribute
        name on the parent.

        This attribute is a so-called "memoized" property.  It initializes
        itself with a new ``weakref.WeakKeyDictionary`` the first time
        it is accessed, returning the same object upon subsequent access.

        .. versionchanged:: 1.4 the :class:`.InstanceState` is now used
           as the key in the weak dictionary rather than the instance
           itself.

        """
        return weakref.WeakKeyDictionary()

    @classmethod
    def coerce(cls, key: str, value: Any) -> Optional[Any]:
        """Given a value, coerce it into the target type.

        Can be overridden by custom subclasses to coerce incoming
        data into a particular type.

        By default, raises ``ValueError``.

        This method is called in different scenarios depending on if
        the parent class is of type :class:`.Mutable` or of type
        :class:`.MutableComposite`.  In the case of the former, it is called
        for both attribute-set operations as well as during ORM loading
        operations.  For the latter, it is only called during attribute-set
        operations; the mechanics of the :func:`.composite` construct
        handle coercion during load operations.


        :param key: string name of the ORM-mapped attribute being set.
        :param value: the incoming value.
        :return: the method should return the coerced value, or raise
         ``ValueError`` if the coercion cannot be completed.

        """
        if value is None:
            return None
        msg = "Attribute '%s' does not accept objects of type %s"
        raise ValueError(msg % (key, type(value)))

    @classmethod
    def _get_listen_keys(cls, attribute: QueryableAttribute[Any]) -> Set[str]:
        """Given a descriptor attribute, return a ``set()`` of the attribute
        keys which indicate a change in the state of this attribute.

        This is normally just ``set([attribute.key])``, but can be overridden
        to provide for additional keys.  E.g. a :class:`.MutableComposite`
        augments this set with the attribute keys associated with the columns
        that comprise the composite value.

        This collection is consulted in the case of intercepting the
        :meth:`.InstanceEvents.refresh` and
        :meth:`.InstanceEvents.refresh_flush` events, which pass along a list
        of attribute names that have been refreshed; the list is compared
        against this set to determine if action needs to be taken.

        """
        return {attribute.key}

    @classmethod
    def _listen_on_attribute(cls, attribute: QueryableAttribute[Any], coerce: bool, parent_cls: _ExternalEntityType[Any]) -> None:
        """Establish this type as a mutation listener for the given
        mapped descriptor.

        """
        key = attribute.key
        if parent_cls is not attribute.class_:
            return
        parent_cls = attribute.class_
        listen_keys = cls._get_listen_keys(attribute)

        def load(state: InstanceState[_O], *args: Any) -> None:
            """Listen for objects loaded or refreshed.

            Wrap the target data member's value with
            ``Mutable``.

            """
            val = state.dict.get(key, None)
            if val is not None:
                if coerce:
                    val = cls.coerce(key, val)
                    state.dict[key] = val
                val._parents[state] = key

        def load_attrs(state: InstanceState[_O], ctx: Union[object, QueryContext, UOWTransaction], attrs: Iterable[Any]) -> None:
            if not attrs or listen_keys.intersection(attrs):
                load(state)

        def set_(target: InstanceState[_O], value: MutableBase | None, oldvalue: MutableBase | None, initiator: AttributeEventToken) -> MutableBase | None:
            """Listen for set/replace events on the target
            data member.

            Establish a weak reference to the parent object
            on the incoming value, remove it for the one
            outgoing.

            """
            if value is oldvalue:
                return value
            if not isinstance(value, cls):
                value = cls.coerce(key, value)
            if value is not None:
                value._parents[target] = key
            if isinstance(oldvalue, cls):
                oldvalue._parents.pop(inspect(target), None)
            return value

        def pickle(state: InstanceState[_O], state_dict: Dict[str, Any]) -> None:
            val = state.dict.get(key, None)
            if val is not None:
                if 'ext.mutable.values' not in state_dict:
                    state_dict['ext.mutable.values'] = defaultdict(list)
                state_dict['ext.mutable.values'][key].append(val)

        def unpickle(state: InstanceState[_O], state_dict: Dict[str, Any]) -> None:
            if 'ext.mutable.values' in state_dict:
                collection = state_dict['ext.mutable.values']
                if isinstance(collection, list):
                    for val in collection:
                        val._parents[state] = key
                else:
                    for val in state_dict['ext.mutable.values'][key]:
                        val._parents[state] = key
        event.listen(parent_cls, '_sa_event_merge_wo_load', load, raw=True, propagate=True)
        event.listen(parent_cls, 'load', load, raw=True, propagate=True)
        event.listen(parent_cls, 'refresh', load_attrs, raw=True, propagate=True)
        event.listen(parent_cls, 'refresh_flush', load_attrs, raw=True, propagate=True)
        event.listen(attribute, 'set', set_, raw=True, retval=True, propagate=True)
        event.listen(parent_cls, 'pickle', pickle, raw=True, propagate=True)
        event.listen(parent_cls, 'unpickle', unpickle, raw=True, propagate=True)