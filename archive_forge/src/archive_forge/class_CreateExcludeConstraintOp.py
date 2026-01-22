from __future__ import annotations
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import Numeric
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import BIGINT
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import NULLTYPE
from .base import alter_column
from .base import alter_table
from .base import AlterColumn
from .base import ColumnComment
from .base import format_column_name
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import ComparisonResult
from .impl import DefaultImpl
from .. import util
from ..autogenerate import render
from ..operations import ops
from ..operations import schemaobj
from ..operations.base import BatchOperations
from ..operations.base import Operations
from ..util import sqla_compat
from ..util.sqla_compat import compiles
@Operations.register_operation('create_exclude_constraint')
@BatchOperations.register_operation('create_exclude_constraint', 'batch_create_exclude_constraint')
@ops.AddConstraintOp.register_add_constraint('exclude_constraint')
class CreateExcludeConstraintOp(ops.AddConstraintOp):
    """Represent a create exclude constraint operation."""
    constraint_type = 'exclude'

    def __init__(self, constraint_name: sqla_compat._ConstraintName, table_name: Union[str, quoted_name], elements: Union[Sequence[Tuple[str, str]], Sequence[Tuple[ColumnClause[Any], str]]], where: Optional[Union[ColumnElement[bool], str]]=None, schema: Optional[str]=None, _orig_constraint: Optional[ExcludeConstraint]=None, **kw) -> None:
        self.constraint_name = constraint_name
        self.table_name = table_name
        self.elements = elements
        self.where = where
        self.schema = schema
        self._orig_constraint = _orig_constraint
        self.kw = kw

    @classmethod
    def from_constraint(cls, constraint: ExcludeConstraint) -> CreateExcludeConstraintOp:
        constraint_table = sqla_compat._table_for_constraint(constraint)
        return cls(constraint.name, constraint_table.name, [(expr, op) for expr, name, op in constraint._render_exprs], where=cast('ColumnElement[bool] | None', constraint.where), schema=constraint_table.schema, _orig_constraint=constraint, deferrable=constraint.deferrable, initially=constraint.initially, using=constraint.using)

    def to_constraint(self, migration_context: Optional[MigrationContext]=None) -> ExcludeConstraint:
        if self._orig_constraint is not None:
            return self._orig_constraint
        schema_obj = schemaobj.SchemaObjects(migration_context)
        t = schema_obj.table(self.table_name, schema=self.schema)
        excl = ExcludeConstraint(*self.elements, name=self.constraint_name, where=self.where, **self.kw)
        for expr, name, oper in excl._render_exprs:
            t.append_column(Column(name, NULLTYPE))
        t.append_constraint(excl)
        return excl

    @classmethod
    def create_exclude_constraint(cls, operations: Operations, constraint_name: str, table_name: str, *elements: Any, **kw: Any) -> Optional[Table]:
        """Issue an alter to create an EXCLUDE constraint using the
        current migration context.

        .. note::  This method is Postgresql specific, and additionally
           requires at least SQLAlchemy 1.0.

        e.g.::

            from alembic import op

            op.create_exclude_constraint(
                "user_excl",
                "user",
                ("period", "&&"),
                ("group", "="),
                where=("group != 'some group'"),
            )

        Note that the expressions work the same way as that of
        the ``ExcludeConstraint`` object itself; if plain strings are
        passed, quoting rules must be applied manually.

        :param name: Name of the constraint.
        :param table_name: String name of the source table.
        :param elements: exclude conditions.
        :param where: SQL expression or SQL string with optional WHERE
         clause.
        :param deferrable: optional bool. If set, emit DEFERRABLE or
         NOT DEFERRABLE when issuing DDL for this constraint.
        :param initially: optional string. If set, emit INITIALLY <value>
         when issuing DDL for this constraint.
        :param schema: Optional schema name to operate within.

        """
        op = cls(constraint_name, table_name, elements, **kw)
        return operations.invoke(op)

    @classmethod
    def batch_create_exclude_constraint(cls, operations: BatchOperations, constraint_name: str, *elements: Any, **kw: Any) -> Optional[Table]:
        """Issue a "create exclude constraint" instruction using the
        current batch migration context.

        .. note::  This method is Postgresql specific, and additionally
           requires at least SQLAlchemy 1.0.

        .. seealso::

            :meth:`.Operations.create_exclude_constraint`

        """
        kw['schema'] = operations.impl.schema
        op = cls(constraint_name, operations.impl.table_name, elements, **kw)
        return operations.invoke(op)