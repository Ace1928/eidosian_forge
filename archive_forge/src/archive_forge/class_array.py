from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TypeVar
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import OVERLAP
from ... import types as sqltypes
from ... import util
from ...sql import expression
from ...sql import operators
from ...sql._typing import _TypeEngineArgument
class array(expression.ExpressionClauseList[_T]):
    """A PostgreSQL ARRAY literal.

    This is used to produce ARRAY literals in SQL expressions, e.g.::

        from sqlalchemy.dialects.postgresql import array
        from sqlalchemy.dialects import postgresql
        from sqlalchemy import select, func

        stmt = select(array([1,2]) + array([3,4,5]))

        print(stmt.compile(dialect=postgresql.dialect()))

    Produces the SQL::

        SELECT ARRAY[%(param_1)s, %(param_2)s] ||
            ARRAY[%(param_3)s, %(param_4)s, %(param_5)s]) AS anon_1

    An instance of :class:`.array` will always have the datatype
    :class:`_types.ARRAY`.  The "inner" type of the array is inferred from
    the values present, unless the ``type_`` keyword argument is passed::

        array(['foo', 'bar'], type_=CHAR)

    Multidimensional arrays are produced by nesting :class:`.array` constructs.
    The dimensionality of the final :class:`_types.ARRAY`
    type is calculated by
    recursively adding the dimensions of the inner :class:`_types.ARRAY`
    type::

        stmt = select(
            array([
                array([1, 2]), array([3, 4]), array([column('q'), column('x')])
            ])
        )
        print(stmt.compile(dialect=postgresql.dialect()))

    Produces::

        SELECT ARRAY[ARRAY[%(param_1)s, %(param_2)s],
        ARRAY[%(param_3)s, %(param_4)s], ARRAY[q, x]] AS anon_1

    .. versionadded:: 1.3.6 added support for multidimensional array literals

    .. seealso::

        :class:`_postgresql.ARRAY`

    """
    __visit_name__ = 'array'
    stringify_dialect = 'postgresql'
    inherit_cache = True

    def __init__(self, clauses, **kw):
        type_arg = kw.pop('type_', None)
        super().__init__(operators.comma_op, *clauses, **kw)
        self._type_tuple = [arg.type for arg in self.clauses]
        main_type = type_arg if type_arg is not None else self._type_tuple[0] if self._type_tuple else sqltypes.NULLTYPE
        if isinstance(main_type, ARRAY):
            self.type = ARRAY(main_type.item_type, dimensions=main_type.dimensions + 1 if main_type.dimensions is not None else 2)
        else:
            self.type = ARRAY(main_type)

    @property
    def _select_iterable(self):
        return (self,)

    def _bind_param(self, operator, obj, _assume_scalar=False, type_=None):
        if _assume_scalar or operator is operators.getitem:
            return expression.BindParameter(None, obj, _compared_to_operator=operator, type_=type_, _compared_to_type=self.type, unique=True)
        else:
            return array([self._bind_param(operator, o, _assume_scalar=True, type_=type_) for o in obj])

    def self_group(self, against=None):
        if against in (operators.any_op, operators.all_op, operators.getitem):
            return expression.Grouping(self)
        else:
            return self