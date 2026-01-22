from __future__ import annotations
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import relationships
from . import util as orm_util
from .base import PassiveFlag
from .query import Query
from .session import object_session
from .writeonly import AbstractCollectionWriter
from .writeonly import WriteOnlyAttributeImpl
from .writeonly import WriteOnlyHistory
from .writeonly import WriteOnlyLoader
from .. import util
from ..engine import result
class DynamicAttributeImpl(WriteOnlyAttributeImpl):
    _supports_dynamic_iteration = True
    collection_history_cls = DynamicCollectionHistory[Any]
    query_class: Type[AppenderMixin[Any]]

    def __init__(self, class_: Union[Type[Any], AliasedClass[Any]], key: str, dispatch: _Dispatch[QueryableAttribute[Any]], target_mapper: Mapper[_T], order_by: _RelationshipOrderByArg, query_class: Optional[Type[AppenderMixin[_T]]]=None, **kw: Any) -> None:
        attributes.AttributeImpl.__init__(self, class_, key, None, dispatch, **kw)
        self.target_mapper = target_mapper
        if order_by:
            self.order_by = tuple(order_by)
        if not query_class:
            self.query_class = AppenderQuery
        elif AppenderMixin in query_class.mro():
            self.query_class = query_class
        else:
            self.query_class = mixin_user_query(query_class)