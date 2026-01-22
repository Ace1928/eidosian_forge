from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
@Operations.register_operation('drop_column')
@BatchOperations.register_operation('drop_column', 'batch_drop_column')
class DropColumnOp(AlterTableOp):
    """Represent a drop column operation."""

    def __init__(self, table_name: str, column_name: str, *, schema: Optional[str]=None, _reverse: Optional[AddColumnOp]=None, **kw: Any) -> None:
        super().__init__(table_name, schema=schema)
        self.column_name = column_name
        self.kw = kw
        self._reverse = _reverse

    def to_diff_tuple(self) -> Tuple[str, Optional[str], str, Column[Any]]:
        return ('remove_column', self.schema, self.table_name, self.to_column())

    def reverse(self) -> AddColumnOp:
        if self._reverse is None:
            raise ValueError('operation is not reversible; original column is not present')
        return AddColumnOp.from_column_and_tablename(self.schema, self.table_name, self._reverse.column)

    @classmethod
    def from_column_and_tablename(cls, schema: Optional[str], tname: str, col: Column[Any]) -> DropColumnOp:
        return cls(tname, col.name, schema=schema, _reverse=AddColumnOp.from_column_and_tablename(schema, tname, col))

    def to_column(self, migration_context: Optional[MigrationContext]=None) -> Column[Any]:
        if self._reverse is not None:
            return self._reverse.column
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.column(self.column_name, NULLTYPE)

    @classmethod
    def drop_column(cls, operations: Operations, table_name: str, column_name: str, *, schema: Optional[str]=None, **kw: Any) -> None:
        """Issue a "drop column" instruction using the current
        migration context.

        e.g.::

            drop_column("organization", "account_id")

        :param table_name: name of table
        :param column_name: name of column
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.
        :param mssql_drop_check: Optional boolean.  When ``True``, on
         Microsoft SQL Server only, first
         drop the CHECK constraint on the column using a
         SQL-script-compatible
         block that selects into a @variable from sys.check_constraints,
         then exec's a separate DROP CONSTRAINT for that constraint.
        :param mssql_drop_default: Optional boolean.  When ``True``, on
         Microsoft SQL Server only, first
         drop the DEFAULT constraint on the column using a
         SQL-script-compatible
         block that selects into a @variable from sys.default_constraints,
         then exec's a separate DROP CONSTRAINT for that default.
        :param mssql_drop_foreign_key: Optional boolean.  When ``True``, on
         Microsoft SQL Server only, first
         drop a single FOREIGN KEY constraint on the column using a
         SQL-script-compatible
         block that selects into a @variable from
         sys.foreign_keys/sys.foreign_key_columns,
         then exec's a separate DROP CONSTRAINT for that default.  Only
         works if the column has exactly one FK constraint which refers to
         it, at the moment.

        """
        op = cls(table_name, column_name, schema=schema, **kw)
        return operations.invoke(op)

    @classmethod
    def batch_drop_column(cls, operations: BatchOperations, column_name: str, **kw: Any) -> None:
        """Issue a "drop column" instruction using the current
        batch migration context.

        .. seealso::

            :meth:`.Operations.drop_column`

        """
        op = cls(operations.impl.table_name, column_name, schema=operations.impl.schema, **kw)
        return operations.invoke(op)