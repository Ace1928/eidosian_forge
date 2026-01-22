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
@Operations.register_operation('create_table')
class CreateTableOp(MigrateOperation):
    """Represent a create table operation."""

    def __init__(self, table_name: str, columns: Sequence[SchemaItem], *, schema: Optional[str]=None, _namespace_metadata: Optional[MetaData]=None, _constraints_included: bool=False, **kw: Any) -> None:
        self.table_name = table_name
        self.columns = columns
        self.schema = schema
        self.info = kw.pop('info', {})
        self.comment = kw.pop('comment', None)
        self.prefixes = kw.pop('prefixes', None)
        self.kw = kw
        self._namespace_metadata = _namespace_metadata
        self._constraints_included = _constraints_included

    def reverse(self) -> DropTableOp:
        return DropTableOp.from_table(self.to_table(), _namespace_metadata=self._namespace_metadata)

    def to_diff_tuple(self) -> Tuple[str, Table]:
        return ('add_table', self.to_table())

    @classmethod
    def from_table(cls, table: Table, *, _namespace_metadata: Optional[MetaData]=None) -> CreateTableOp:
        if _namespace_metadata is None:
            _namespace_metadata = table.metadata
        return cls(table.name, list(table.c) + list(table.constraints), schema=table.schema, _namespace_metadata=_namespace_metadata, _constraints_included=True, comment=table.comment, info=dict(table.info), prefixes=list(table._prefixes), **table.kwargs)

    def to_table(self, migration_context: Optional[MigrationContext]=None) -> Table:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.table(self.table_name, *self.columns, schema=self.schema, prefixes=list(self.prefixes) if self.prefixes else [], comment=self.comment, info=self.info.copy() if self.info else {}, _constraints_included=self._constraints_included, **self.kw)

    @classmethod
    def create_table(cls, operations: Operations, table_name: str, *columns: SchemaItem, **kw: Any) -> Table:
        """Issue a "create table" instruction using the current migration
        context.

        This directive receives an argument list similar to that of the
        traditional :class:`sqlalchemy.schema.Table` construct, but without the
        metadata::

            from sqlalchemy import INTEGER, VARCHAR, NVARCHAR, Column
            from alembic import op

            op.create_table(
                "account",
                Column("id", INTEGER, primary_key=True),
                Column("name", VARCHAR(50), nullable=False),
                Column("description", NVARCHAR(200)),
                Column("timestamp", TIMESTAMP, server_default=func.now()),
            )

        Note that :meth:`.create_table` accepts
        :class:`~sqlalchemy.schema.Column`
        constructs directly from the SQLAlchemy library.  In particular,
        default values to be created on the database side are
        specified using the ``server_default`` parameter, and not
        ``default`` which only specifies Python-side defaults::

            from alembic import op
            from sqlalchemy import Column, TIMESTAMP, func

            # specify "DEFAULT NOW" along with the "timestamp" column
            op.create_table(
                "account",
                Column("id", INTEGER, primary_key=True),
                Column("timestamp", TIMESTAMP, server_default=func.now()),
            )

        The function also returns a newly created
        :class:`~sqlalchemy.schema.Table` object, corresponding to the table
        specification given, which is suitable for
        immediate SQL operations, in particular
        :meth:`.Operations.bulk_insert`::

            from sqlalchemy import INTEGER, VARCHAR, NVARCHAR, Column
            from alembic import op

            account_table = op.create_table(
                "account",
                Column("id", INTEGER, primary_key=True),
                Column("name", VARCHAR(50), nullable=False),
                Column("description", NVARCHAR(200)),
                Column("timestamp", TIMESTAMP, server_default=func.now()),
            )

            op.bulk_insert(
                account_table,
                [
                    {"name": "A1", "description": "account 1"},
                    {"name": "A2", "description": "account 2"},
                ],
            )

        :param table_name: Name of the table
        :param \\*columns: collection of :class:`~sqlalchemy.schema.Column`
         objects within
         the table, as well as optional :class:`~sqlalchemy.schema.Constraint`
         objects
         and :class:`~.sqlalchemy.schema.Index` objects.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.
        :param \\**kw: Other keyword arguments are passed to the underlying
         :class:`sqlalchemy.schema.Table` object created for the command.

        :return: the :class:`~sqlalchemy.schema.Table` object corresponding
         to the parameters given.

        """
        op = cls(table_name, columns, **kw)
        return operations.invoke(op)