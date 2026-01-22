from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
from . import types
from .array import ARRAY
from ...sql import coercions
from ...sql import elements
from ...sql import expression
from ...sql import functions
from ...sql import roles
from ...sql import schema
from ...sql.schema import ColumnCollectionConstraint
from ...sql.sqltypes import TEXT
from ...sql.visitors import InternalTraversal
class aggregate_order_by(expression.ColumnElement):
    """Represent a PostgreSQL aggregate order by expression.

    E.g.::

        from sqlalchemy.dialects.postgresql import aggregate_order_by
        expr = func.array_agg(aggregate_order_by(table.c.a, table.c.b.desc()))
        stmt = select(expr)

    would represent the expression::

        SELECT array_agg(a ORDER BY b DESC) FROM table;

    Similarly::

        expr = func.string_agg(
            table.c.a,
            aggregate_order_by(literal_column("','"), table.c.a)
        )
        stmt = select(expr)

    Would represent::

        SELECT string_agg(a, ',' ORDER BY a) FROM table;

    .. versionchanged:: 1.2.13 - the ORDER BY argument may be multiple terms

    .. seealso::

        :class:`_functions.array_agg`

    """
    __visit_name__ = 'aggregate_order_by'
    stringify_dialect = 'postgresql'
    _traverse_internals: _TraverseInternalsType = [('target', InternalTraversal.dp_clauseelement), ('type', InternalTraversal.dp_type), ('order_by', InternalTraversal.dp_clauseelement)]

    def __init__(self, target, *order_by):
        self.target = coercions.expect(roles.ExpressionElementRole, target)
        self.type = self.target.type
        _lob = len(order_by)
        if _lob == 0:
            raise TypeError('at least one ORDER BY element is required')
        elif _lob == 1:
            self.order_by = coercions.expect(roles.ExpressionElementRole, order_by[0])
        else:
            self.order_by = elements.ClauseList(*order_by, _literal_as_text_role=roles.ExpressionElementRole)

    def self_group(self, against=None):
        return self

    def get_children(self, **kwargs):
        return (self.target, self.order_by)

    def _copy_internals(self, clone=elements._clone, **kw):
        self.target = clone(self.target, **kw)
        self.order_by = clone(self.order_by, **kw)

    @property
    def _from_objects(self):
        return self.target._from_objects + self.order_by._from_objects