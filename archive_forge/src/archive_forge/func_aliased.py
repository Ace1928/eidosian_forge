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
def aliased(element: Union[_EntityType[_O], FromClause], alias: Optional[FromClause]=None, name: Optional[str]=None, flat: bool=False, adapt_on_names: bool=False) -> Union[AliasedClass[_O], FromClause, AliasedType[_O]]:
    """Produce an alias of the given element, usually an :class:`.AliasedClass`
    instance.

    E.g.::

        my_alias = aliased(MyClass)

        stmt = select(MyClass, my_alias).filter(MyClass.id > my_alias.id)
        result = session.execute(stmt)

    The :func:`.aliased` function is used to create an ad-hoc mapping of a
    mapped class to a new selectable.  By default, a selectable is generated
    from the normally mapped selectable (typically a :class:`_schema.Table`
    ) using the
    :meth:`_expression.FromClause.alias` method. However, :func:`.aliased`
    can also be
    used to link the class to a new :func:`_expression.select` statement.
    Also, the :func:`.with_polymorphic` function is a variant of
    :func:`.aliased` that is intended to specify a so-called "polymorphic
    selectable", that corresponds to the union of several joined-inheritance
    subclasses at once.

    For convenience, the :func:`.aliased` function also accepts plain
    :class:`_expression.FromClause` constructs, such as a
    :class:`_schema.Table` or
    :func:`_expression.select` construct.   In those cases, the
    :meth:`_expression.FromClause.alias`
    method is called on the object and the new
    :class:`_expression.Alias` object returned.  The returned
    :class:`_expression.Alias` is not
    ORM-mapped in this case.

    .. seealso::

        :ref:`tutorial_orm_entity_aliases` - in the :ref:`unified_tutorial`

        :ref:`orm_queryguide_orm_aliases` - in the :ref:`queryguide_toplevel`

    :param element: element to be aliased.  Is normally a mapped class,
     but for convenience can also be a :class:`_expression.FromClause`
     element.

    :param alias: Optional selectable unit to map the element to.  This is
     usually used to link the object to a subquery, and should be an aliased
     select construct as one would produce from the
     :meth:`_query.Query.subquery` method or
     the :meth:`_expression.Select.subquery` or
     :meth:`_expression.Select.alias` methods of the :func:`_expression.select`
     construct.

    :param name: optional string name to use for the alias, if not specified
     by the ``alias`` parameter.  The name, among other things, forms the
     attribute name that will be accessible via tuples returned by a
     :class:`_query.Query` object.  Not supported when creating aliases
     of :class:`_sql.Join` objects.

    :param flat: Boolean, will be passed through to the
     :meth:`_expression.FromClause.alias` call so that aliases of
     :class:`_expression.Join` objects will alias the individual tables
     inside the join, rather than creating a subquery.  This is generally
     supported by all modern databases with regards to right-nested joins
     and generally produces more efficient queries.

    :param adapt_on_names: if True, more liberal "matching" will be used when
     mapping the mapped columns of the ORM entity to those of the
     given selectable - a name-based match will be performed if the
     given selectable doesn't otherwise have a column that corresponds
     to one on the entity.  The use case for this is when associating
     an entity with some derived selectable such as one that uses
     aggregate functions::

        class UnitPrice(Base):
            __tablename__ = 'unit_price'
            ...
            unit_id = Column(Integer)
            price = Column(Numeric)

        aggregated_unit_price = Session.query(
                                    func.sum(UnitPrice.price).label('price')
                                ).group_by(UnitPrice.unit_id).subquery()

        aggregated_unit_price = aliased(UnitPrice,
                    alias=aggregated_unit_price, adapt_on_names=True)

     Above, functions on ``aggregated_unit_price`` which refer to
     ``.price`` will return the
     ``func.sum(UnitPrice.price).label('price')`` column, as it is
     matched on the name "price".  Ordinarily, the "price" function
     wouldn't have any "column correspondence" to the actual
     ``UnitPrice.price`` column as it is not a proxy of the original.

    """
    return AliasedInsp._alias_factory(element, alias=alias, name=name, flat=flat, adapt_on_names=adapt_on_names)