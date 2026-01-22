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
@Operations.register_operation('create_primary_key')
@BatchOperations.register_operation('create_primary_key', 'batch_create_primary_key')
@AddConstraintOp.register_add_constraint('primary_key_constraint')
class CreatePrimaryKeyOp(AddConstraintOp):
    """Represent a create primary key operation."""
    constraint_type = 'primarykey'

    def __init__(self, constraint_name: Optional[sqla_compat._ConstraintNameDefined], table_name: str, columns: Sequence[str], *, schema: Optional[str]=None, **kw: Any) -> None:
        self.constraint_name = constraint_name
        self.table_name = table_name
        self.columns = columns
        self.schema = schema
        self.kw = kw

    @classmethod
    def from_constraint(cls, constraint: Constraint) -> CreatePrimaryKeyOp:
        constraint_table = sqla_compat._table_for_constraint(constraint)
        pk_constraint = cast('PrimaryKeyConstraint', constraint)
        return cls(sqla_compat.constraint_name_or_none(pk_constraint.name), constraint_table.name, pk_constraint.columns.keys(), schema=constraint_table.schema, **pk_constraint.dialect_kwargs)

    def to_constraint(self, migration_context: Optional[MigrationContext]=None) -> PrimaryKeyConstraint:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.primary_key_constraint(self.constraint_name, self.table_name, self.columns, schema=self.schema, **self.kw)

    @classmethod
    def create_primary_key(cls, operations: Operations, constraint_name: Optional[str], table_name: str, columns: List[str], *, schema: Optional[str]=None) -> None:
        """Issue a "create primary key" instruction using the current
        migration context.

        e.g.::

            from alembic import op

            op.create_primary_key("pk_my_table", "my_table", ["id", "version"])

        This internally generates a :class:`~sqlalchemy.schema.Table` object
        containing the necessary columns, then generates a new
        :class:`~sqlalchemy.schema.PrimaryKeyConstraint`
        object which it then associates with the
        :class:`~sqlalchemy.schema.Table`.
        Any event listeners associated with this action will be fired
        off normally.   The :class:`~sqlalchemy.schema.AddConstraint`
        construct is ultimately used to generate the ALTER statement.

        :param constraint_name: Name of the primary key constraint.  The name
         is necessary so that an ALTER statement can be emitted.  For setups
         that use an automated naming scheme such as that described at
         :ref:`sqla:constraint_naming_conventions`
         ``name`` here can be ``None``, as the event listener will
         apply the name to the constraint object when it is associated
         with the table.
        :param table_name: String name of the target table.
        :param columns: a list of string column names to be applied to the
         primary key constraint.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.

        """
        op = cls(constraint_name, table_name, columns, schema=schema)
        return operations.invoke(op)

    @classmethod
    def batch_create_primary_key(cls, operations: BatchOperations, constraint_name: str, columns: List[str]) -> None:
        """Issue a "create primary key" instruction using the
        current batch migration context.

        The batch form of this call omits the ``table_name`` and ``schema``
        arguments from the call.

        .. seealso::

            :meth:`.Operations.create_primary_key`

        """
        op = cls(constraint_name, operations.impl.table_name, columns, schema=operations.impl.schema)
        return operations.invoke(op)