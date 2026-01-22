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
class AppenderMixin(AbstractCollectionWriter[_T]):
    """A mixin that expects to be mixing in a Query class with
    AbstractAppender.


    """
    query_class: Optional[Type[Query[_T]]] = None
    _order_by_clauses: Tuple[ColumnElement[Any], ...]

    def __init__(self, attr: DynamicAttributeImpl, state: InstanceState[_T]) -> None:
        Query.__init__(self, attr.target_mapper, None)
        super().__init__(attr, state)

    @property
    def session(self) -> Optional[Session]:
        sess = object_session(self.instance)
        if sess is not None and sess.autoflush and (self.instance in sess):
            sess.flush()
        if not orm_util.has_identity(self.instance):
            return None
        else:
            return sess

    @session.setter
    def session(self, session: Session) -> None:
        self.sess = session

    def _iter(self) -> Union[result.ScalarResult[_T], result.Result[_T]]:
        sess = self.session
        if sess is None:
            state = attributes.instance_state(self.instance)
            if state.detached:
                util.warn('Instance %s is detached, dynamic relationship cannot return a correct result.   This warning will become a DetachedInstanceError in a future release.' % orm_util.state_str(state))
            return result.IteratorResult(result.SimpleResultMetaData([self.attr.class_.__name__]), self.attr._get_collection_history(attributes.instance_state(self.instance), PassiveFlag.PASSIVE_NO_INITIALIZE).added_items, _source_supports_scalars=True).scalars()
        else:
            return self._generate(sess)._iter()
    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[_T]:
            ...

    def __getitem__(self, index: Any) -> Union[_T, List[_T]]:
        sess = self.session
        if sess is None:
            return self.attr._get_collection_history(attributes.instance_state(self.instance), PassiveFlag.PASSIVE_NO_INITIALIZE).indexed(index)
        else:
            return self._generate(sess).__getitem__(index)

    def count(self) -> int:
        sess = self.session
        if sess is None:
            return len(self.attr._get_collection_history(attributes.instance_state(self.instance), PassiveFlag.PASSIVE_NO_INITIALIZE).added_items)
        else:
            return self._generate(sess).count()

    def _generate(self, sess: Optional[Session]=None) -> Query[_T]:
        instance = self.instance
        if sess is None:
            sess = object_session(instance)
            if sess is None:
                raise orm_exc.DetachedInstanceError("Parent instance %s is not bound to a Session, and no contextual session is established; lazy load operation of attribute '%s' cannot proceed" % (orm_util.instance_str(instance), self.attr.key))
        if self.query_class:
            query = self.query_class(self.attr.target_mapper, session=sess)
        else:
            query = sess.query(self.attr.target_mapper)
        query._where_criteria = self._where_criteria
        query._from_obj = self._from_obj
        query._order_by_clauses = self._order_by_clauses
        return query

    def add_all(self, iterator: Iterable[_T]) -> None:
        """Add an iterable of items to this :class:`_orm.AppenderQuery`.

        The given items will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        This method is provided to assist in delivering forwards-compatibility
        with the :class:`_orm.WriteOnlyCollection` collection class.

        .. versionadded:: 2.0

        """
        self._add_all_impl(iterator)

    def add(self, item: _T) -> None:
        """Add an item to this :class:`_orm.AppenderQuery`.

        The given item will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        This method is provided to assist in delivering forwards-compatibility
        with the :class:`_orm.WriteOnlyCollection` collection class.

        .. versionadded:: 2.0

        """
        self._add_all_impl([item])

    def extend(self, iterator: Iterable[_T]) -> None:
        """Add an iterable of items to this :class:`_orm.AppenderQuery`.

        The given items will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        """
        self._add_all_impl(iterator)

    def append(self, item: _T) -> None:
        """Append an item to this :class:`_orm.AppenderQuery`.

        The given item will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        """
        self._add_all_impl([item])

    def remove(self, item: _T) -> None:
        """Remove an item from this :class:`_orm.AppenderQuery`.

        The given item will be removed from the parent instance's collection on
        the next flush.

        """
        self._remove_impl(item)