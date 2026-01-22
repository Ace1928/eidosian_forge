from __future__ import annotations
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql import bindparam
from . import attributes
from . import interfaces
from . import relationships
from . import strategies
from .base import NEVER_SET
from .base import object_mapper
from .base import PassiveFlag
from .base import RelationshipDirection
from .. import exc
from .. import inspect
from .. import log
from .. import util
from ..sql import delete
from ..sql import insert
from ..sql import select
from ..sql import update
from ..sql.dml import Delete
from ..sql.dml import Insert
from ..sql.dml import Update
from ..util.typing import Literal
class WriteOnlyAttributeImpl(attributes.HasCollectionAdapter, attributes.AttributeImpl):
    uses_objects: bool = True
    default_accepts_scalar_loader: bool = False
    supports_population: bool = False
    _supports_dynamic_iteration: bool = False
    collection: bool = False
    dynamic: bool = True
    order_by: _RelationshipOrderByArg = ()
    collection_history_cls: Type[WriteOnlyHistory[Any]] = WriteOnlyHistory
    query_class: Type[WriteOnlyCollection[Any]]

    def __init__(self, class_: Union[Type[Any], AliasedClass[Any]], key: str, dispatch: _Dispatch[QueryableAttribute[Any]], target_mapper: Mapper[_T], order_by: _RelationshipOrderByArg, **kw: Any):
        super().__init__(class_, key, None, dispatch, **kw)
        self.target_mapper = target_mapper
        self.query_class = WriteOnlyCollection
        if order_by:
            self.order_by = tuple(order_by)

    def get(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF) -> Union[util.OrderedIdentitySet, WriteOnlyCollection[Any]]:
        if not passive & PassiveFlag.SQL_OK:
            return self._get_collection_history(state, PassiveFlag.PASSIVE_NO_INITIALIZE).added_items
        else:
            return self.query_class(self, state)

    @overload
    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: Literal[None]=..., passive: Literal[PassiveFlag.PASSIVE_OFF]=...) -> CollectionAdapter:
        ...

    @overload
    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: _AdaptedCollectionProtocol=..., passive: PassiveFlag=...) -> CollectionAdapter:
        ...

    @overload
    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: Optional[_AdaptedCollectionProtocol]=..., passive: PassiveFlag=...) -> Union[Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter]:
        ...

    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: Optional[_AdaptedCollectionProtocol]=None, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF) -> Union[Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter]:
        data: Collection[Any]
        if not passive & PassiveFlag.SQL_OK:
            data = self._get_collection_history(state, passive).added_items
        else:
            history = self._get_collection_history(state, passive)
            data = history.added_plus_unchanged
        return DynamicCollectionAdapter(data)

    @util.memoized_property
    def _append_token(self) -> attributes.AttributeEventToken:
        return attributes.AttributeEventToken(self, attributes.OP_APPEND)

    @util.memoized_property
    def _remove_token(self) -> attributes.AttributeEventToken:
        return attributes.AttributeEventToken(self, attributes.OP_REMOVE)

    def fire_append_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], collection_history: Optional[WriteOnlyHistory[Any]]=None) -> None:
        if collection_history is None:
            collection_history = self._modified_event(state, dict_)
        collection_history.add_added(value)
        for fn in self.dispatch.append:
            value = fn(state, value, initiator or self._append_token)
        if self.trackparent and value is not None:
            self.sethasparent(attributes.instance_state(value), state, True)

    def fire_remove_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], collection_history: Optional[WriteOnlyHistory[Any]]=None) -> None:
        if collection_history is None:
            collection_history = self._modified_event(state, dict_)
        collection_history.add_removed(value)
        if self.trackparent and value is not None:
            self.sethasparent(attributes.instance_state(value), state, False)
        for fn in self.dispatch.remove:
            fn(state, value, initiator or self._remove_token)

    def _modified_event(self, state: InstanceState[Any], dict_: _InstanceDict) -> WriteOnlyHistory[Any]:
        if self.key not in state.committed_state:
            state.committed_state[self.key] = self.collection_history_cls(self, state, PassiveFlag.PASSIVE_NO_FETCH)
        state._modified_event(dict_, self, NEVER_SET)
        dict_[self.key] = True
        return state.committed_state[self.key]

    def set(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken]=None, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF, check_old: Any=None, pop: bool=False, _adapt: bool=True) -> None:
        if initiator and initiator.parent_token is self.parent_token:
            return
        if pop and value is None:
            return
        iterable = value
        new_values = list(iterable)
        if state.has_identity:
            if not self._supports_dynamic_iteration:
                raise exc.InvalidRequestError(f'''Collection "{self}" does not support implicit iteration; collection replacement operations can't be used''')
            old_collection = util.IdentitySet(self.get(state, dict_, passive=passive))
        collection_history = self._modified_event(state, dict_)
        if not state.has_identity:
            old_collection = collection_history.added_items
        else:
            old_collection = old_collection.union(collection_history.added_items)
        constants = old_collection.intersection(new_values)
        additions = util.IdentitySet(new_values).difference(constants)
        removals = old_collection.difference(constants)
        for member in new_values:
            if member in additions:
                self.fire_append_event(state, dict_, member, None, collection_history=collection_history)
        for member in removals:
            self.fire_remove_event(state, dict_, member, None, collection_history=collection_history)

    def delete(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError()

    def set_committed_value(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any) -> NoReturn:
        raise NotImplementedError("Dynamic attributes don't support collection population.")

    def get_history(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PassiveFlag.PASSIVE_NO_FETCH) -> attributes.History:
        c = self._get_collection_history(state, passive)
        return c.as_history()

    def get_all_pending(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PassiveFlag.PASSIVE_NO_INITIALIZE) -> List[Tuple[InstanceState[Any], Any]]:
        c = self._get_collection_history(state, passive)
        return [(attributes.instance_state(x), x) for x in c.all_items]

    def _get_collection_history(self, state: InstanceState[Any], passive: PassiveFlag) -> WriteOnlyHistory[Any]:
        c: WriteOnlyHistory[Any]
        if self.key in state.committed_state:
            c = state.committed_state[self.key]
        else:
            c = self.collection_history_cls(self, state, PassiveFlag.PASSIVE_NO_FETCH)
        if state.has_identity and passive & PassiveFlag.INIT_OK:
            return self.collection_history_cls(self, state, passive, apply_to=c)
        else:
            return c

    def append(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PassiveFlag.PASSIVE_NO_FETCH) -> None:
        if initiator is not self:
            self.fire_append_event(state, dict_, value, initiator)

    def remove(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PassiveFlag.PASSIVE_NO_FETCH) -> None:
        if initiator is not self:
            self.fire_remove_event(state, dict_, value, initiator)

    def pop(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PassiveFlag.PASSIVE_NO_FETCH) -> None:
        self.remove(state, dict_, value, initiator, passive=passive)