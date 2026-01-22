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
class ExcludeConstraint(ColumnCollectionConstraint):
    """A table-level EXCLUDE constraint.

    Defines an EXCLUDE constraint as described in the `PostgreSQL
    documentation`__.

    __ https://www.postgresql.org/docs/current/static/sql-createtable.html#SQL-CREATETABLE-EXCLUDE

    """
    __visit_name__ = 'exclude_constraint'
    where = None
    inherit_cache = False
    create_drop_stringify_dialect = 'postgresql'

    @elements._document_text_coercion('where', ':class:`.ExcludeConstraint`', ':paramref:`.ExcludeConstraint.where`')
    def __init__(self, *elements, **kw):
        """
        Create an :class:`.ExcludeConstraint` object.

        E.g.::

            const = ExcludeConstraint(
                (Column('period'), '&&'),
                (Column('group'), '='),
                where=(Column('group') != 'some group'),
                ops={'group': 'my_operator_class'}
            )

        The constraint is normally embedded into the :class:`_schema.Table`
        construct
        directly, or added later using :meth:`.append_constraint`::

            some_table = Table(
                'some_table', metadata,
                Column('id', Integer, primary_key=True),
                Column('period', TSRANGE()),
                Column('group', String)
            )

            some_table.append_constraint(
                ExcludeConstraint(
                    (some_table.c.period, '&&'),
                    (some_table.c.group, '='),
                    where=some_table.c.group != 'some group',
                    name='some_table_excl_const',
                    ops={'group': 'my_operator_class'}
                )
            )

        The exclude constraint defined in this example requires the
        ``btree_gist`` extension, that can be created using the
        command ``CREATE EXTENSION btree_gist;``.

        :param \\*elements:

          A sequence of two tuples of the form ``(column, operator)`` where
          "column" is either a :class:`_schema.Column` object, or a SQL
          expression element (e.g. ``func.int8range(table.from, table.to)``)
          or the name of a column as string, and "operator" is a string
          containing the operator to use (e.g. `"&&"` or `"="`).

          In order to specify a column name when a :class:`_schema.Column`
          object is not available, while ensuring
          that any necessary quoting rules take effect, an ad-hoc
          :class:`_schema.Column` or :func:`_expression.column`
          object should be used.
          The ``column`` may also be a string SQL expression when
          passed as :func:`_expression.literal_column` or
          :func:`_expression.text`

        :param name:
          Optional, the in-database name of this constraint.

        :param deferrable:
          Optional bool.  If set, emit DEFERRABLE or NOT DEFERRABLE when
          issuing DDL for this constraint.

        :param initially:
          Optional string.  If set, emit INITIALLY <value> when issuing DDL
          for this constraint.

        :param using:
          Optional string.  If set, emit USING <index_method> when issuing DDL
          for this constraint. Defaults to 'gist'.

        :param where:
          Optional SQL expression construct or literal SQL string.
          If set, emit WHERE <predicate> when issuing DDL
          for this constraint.

        :param ops:
          Optional dictionary.  Used to define operator classes for the
          elements; works the same way as that of the
          :ref:`postgresql_ops <postgresql_operator_classes>`
          parameter specified to the :class:`_schema.Index` construct.

          .. versionadded:: 1.3.21

          .. seealso::

            :ref:`postgresql_operator_classes` - general description of how
            PostgreSQL operator classes are specified.

        """
        columns = []
        render_exprs = []
        self.operators = {}
        expressions, operators = zip(*elements)
        for (expr, column, strname, add_element), operator in zip(coercions.expect_col_expression_collection(roles.DDLConstraintColumnRole, expressions), operators):
            if add_element is not None:
                columns.append(add_element)
            name = column.name if column is not None else strname
            if name is not None:
                self.operators[name] = operator
            render_exprs.append((expr, name, operator))
        self._render_exprs = render_exprs
        ColumnCollectionConstraint.__init__(self, *columns, name=kw.get('name'), deferrable=kw.get('deferrable'), initially=kw.get('initially'))
        self.using = kw.get('using', 'gist')
        where = kw.get('where')
        if where is not None:
            self.where = coercions.expect(roles.StatementOptionRole, where)
        self.ops = kw.get('ops', {})

    def _set_parent(self, table, **kw):
        super()._set_parent(table)
        self._render_exprs = [(expr if not isinstance(expr, str) else table.c[expr], name, operator) for expr, name, operator in self._render_exprs]

    def _copy(self, target_table=None, **kw):
        elements = [(schema._copy_expression(expr, self.parent, target_table), operator) for expr, _, operator in self._render_exprs]
        c = self.__class__(*elements, name=self.name, deferrable=self.deferrable, initially=self.initially, where=self.where, using=self.using)
        c.dispatch._update(self.dispatch)
        return c