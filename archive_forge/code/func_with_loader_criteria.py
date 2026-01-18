from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import mapperlib as mapperlib
from ._typing import _O
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .interfaces import _AttributeOptions
from .properties import MappedColumn
from .properties import MappedSQLExpression
from .query import AliasOption
from .relationships import _RelationshipArgumentType
from .relationships import _RelationshipDeclared
from .relationships import _RelationshipSecondaryArgument
from .relationships import RelationshipProperty
from .session import Session
from .util import _ORMJoin
from .util import AliasedClass
from .util import AliasedInsp
from .util import LoaderCriteriaOption
from .. import sql
from .. import util
from ..exc import InvalidRequestError
from ..sql._typing import _no_kw
from ..sql.base import _NoArg
from ..sql.base import SchemaEventTarget
from ..sql.schema import _InsertSentinelColumnDefault
from ..sql.schema import SchemaConst
from ..sql.selectable import FromClause
from ..util.typing import Annotated
from ..util.typing import Literal
def with_loader_criteria(entity_or_base: _EntityType[Any], where_criteria: Union[_ColumnExpressionArgument[bool], Callable[[Any], _ColumnExpressionArgument[bool]]], loader_only: bool=False, include_aliases: bool=False, propagate_to_loaders: bool=True, track_closure_variables: bool=True) -> LoaderCriteriaOption:
    """Add additional WHERE criteria to the load for all occurrences of
    a particular entity.

    .. versionadded:: 1.4

    The :func:`_orm.with_loader_criteria` option is intended to add
    limiting criteria to a particular kind of entity in a query,
    **globally**, meaning it will apply to the entity as it appears
    in the SELECT query as well as within any subqueries, join
    conditions, and relationship loads, including both eager and lazy
    loaders, without the need for it to be specified in any particular
    part of the query.    The rendering logic uses the same system used by
    single table inheritance to ensure a certain discriminator is applied
    to a table.

    E.g., using :term:`2.0-style` queries, we can limit the way the
    ``User.addresses`` collection is loaded, regardless of the kind
    of loading used::

        from sqlalchemy.orm import with_loader_criteria

        stmt = select(User).options(
            selectinload(User.addresses),
            with_loader_criteria(Address, Address.email_address != 'foo'))
        )

    Above, the "selectinload" for ``User.addresses`` will apply the
    given filtering criteria to the WHERE clause.

    Another example, where the filtering will be applied to the
    ON clause of the join, in this example using :term:`1.x style`
    queries::

        q = session.query(User).outerjoin(User.addresses).options(
            with_loader_criteria(Address, Address.email_address != 'foo'))
        )

    The primary purpose of :func:`_orm.with_loader_criteria` is to use
    it in the :meth:`_orm.SessionEvents.do_orm_execute` event handler
    to ensure that all occurrences of a particular entity are filtered
    in a certain way, such as filtering for access control roles.    It
    also can be used to apply criteria to relationship loads.  In the
    example below, we can apply a certain set of rules to all queries
    emitted by a particular :class:`_orm.Session`::

        session = Session(bind=engine)

        @event.listens_for("do_orm_execute", session)
        def _add_filtering_criteria(execute_state):

            if (
                execute_state.is_select
                and not execute_state.is_column_load
                and not execute_state.is_relationship_load
            ):
                execute_state.statement = execute_state.statement.options(
                    with_loader_criteria(
                        SecurityRole,
                        lambda cls: cls.role.in_(['some_role']),
                        include_aliases=True
                    )
                )

    In the above example, the :meth:`_orm.SessionEvents.do_orm_execute`
    event will intercept all queries emitted using the
    :class:`_orm.Session`. For those queries which are SELECT statements
    and are not attribute or relationship loads a custom
    :func:`_orm.with_loader_criteria` option is added to the query.    The
    :func:`_orm.with_loader_criteria` option will be used in the given
    statement and will also be automatically propagated to all relationship
    loads that descend from this query.

    The criteria argument given is a ``lambda`` that accepts a ``cls``
    argument.  The given class will expand to include all mapped subclass
    and need not itself be a mapped class.

    .. tip::

       When using :func:`_orm.with_loader_criteria` option in
       conjunction with the :func:`_orm.contains_eager` loader option,
       it's important to note that :func:`_orm.with_loader_criteria` only
       affects the part of the query that determines what SQL is rendered
       in terms of the WHERE and FROM clauses. The
       :func:`_orm.contains_eager` option does not affect the rendering of
       the SELECT statement outside of the columns clause, so does not have
       any interaction with the :func:`_orm.with_loader_criteria` option.
       However, the way things "work" is that :func:`_orm.contains_eager`
       is meant to be used with a query that is already selecting from the
       additional entities in some way, where
       :func:`_orm.with_loader_criteria` can apply it's additional
       criteria.

       In the example below, assuming a mapping relationship as
       ``A -> A.bs -> B``, the given :func:`_orm.with_loader_criteria`
       option will affect the way in which the JOIN is rendered::

            stmt = select(A).join(A.bs).options(
                contains_eager(A.bs),
                with_loader_criteria(B, B.flag == 1)
            )

       Above, the given :func:`_orm.with_loader_criteria` option will
       affect the ON clause of the JOIN that is specified by
       ``.join(A.bs)``, so is applied as expected. The
       :func:`_orm.contains_eager` option has the effect that columns from
       ``B`` are added to the columns clause::

            SELECT
                b.id, b.a_id, b.data, b.flag,
                a.id AS id_1,
                a.data AS data_1
            FROM a JOIN b ON a.id = b.a_id AND b.flag = :flag_1


       The use of the :func:`_orm.contains_eager` option within the above
       statement has no effect on the behavior of the
       :func:`_orm.with_loader_criteria` option. If the
       :func:`_orm.contains_eager` option were omitted, the SQL would be
       the same as regards the FROM and WHERE clauses, where
       :func:`_orm.with_loader_criteria` continues to add its criteria to
       the ON clause of the JOIN. The addition of
       :func:`_orm.contains_eager` only affects the columns clause, in that
       additional columns against ``b`` are added which are then consumed
       by the ORM to produce ``B`` instances.

    .. warning:: The use of a lambda inside of the call to
      :func:`_orm.with_loader_criteria` is only invoked **once per unique
      class**. Custom functions should not be invoked within this lambda.
      See :ref:`engine_lambda_caching` for an overview of the "lambda SQL"
      feature, which is for advanced use only.

    :param entity_or_base: a mapped class, or a class that is a super
     class of a particular set of mapped classes, to which the rule
     will apply.

    :param where_criteria: a Core SQL expression that applies limiting
     criteria.   This may also be a "lambda:" or Python function that
     accepts a target class as an argument, when the given class is
     a base with many different mapped subclasses.

     .. note:: To support pickling, use a module-level Python function to
        produce the SQL expression instead of a lambda or a fixed SQL
        expression, which tend to not be picklable.

    :param include_aliases: if True, apply the rule to :func:`_orm.aliased`
     constructs as well.

    :param propagate_to_loaders: defaults to True, apply to relationship
     loaders such as lazy loaders.   This indicates that the
     option object itself including SQL expression is carried along with
     each loaded instance.  Set to ``False`` to prevent the object from
     being assigned to individual instances.


     .. seealso::

        :ref:`examples_session_orm_events` - includes examples of using
        :func:`_orm.with_loader_criteria`.

        :ref:`do_orm_execute_global_criteria` - basic example on how to
        combine :func:`_orm.with_loader_criteria` with the
        :meth:`_orm.SessionEvents.do_orm_execute` event.

    :param track_closure_variables: when False, closure variables inside
     of a lambda expression will not be used as part of
     any cache key.    This allows more complex expressions to be used
     inside of a lambda expression but requires that the lambda ensures
     it returns the identical SQL every time given a particular class.

     .. versionadded:: 1.4.0b2

    """
    return LoaderCriteriaOption(entity_or_base, where_criteria, loader_only, include_aliases, propagate_to_loaders, track_closure_variables)