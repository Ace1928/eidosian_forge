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
@Operations.register_operation('create_foreign_key')
@BatchOperations.register_operation('create_foreign_key', 'batch_create_foreign_key')
@AddConstraintOp.register_add_constraint('foreign_key_constraint')
class CreateForeignKeyOp(AddConstraintOp):
    """Represent a create foreign key constraint operation."""
    constraint_type = 'foreignkey'

    def __init__(self, constraint_name: Optional[sqla_compat._ConstraintNameDefined], source_table: str, referent_table: str, local_cols: List[str], remote_cols: List[str], **kw: Any) -> None:
        self.constraint_name = constraint_name
        self.source_table = source_table
        self.referent_table = referent_table
        self.local_cols = local_cols
        self.remote_cols = remote_cols
        self.kw = kw

    def to_diff_tuple(self) -> Tuple[str, ForeignKeyConstraint]:
        return ('add_fk', self.to_constraint())

    @classmethod
    def from_constraint(cls, constraint: Constraint) -> CreateForeignKeyOp:
        fk_constraint = cast('ForeignKeyConstraint', constraint)
        kw: Dict[str, Any] = {}
        if fk_constraint.onupdate:
            kw['onupdate'] = fk_constraint.onupdate
        if fk_constraint.ondelete:
            kw['ondelete'] = fk_constraint.ondelete
        if fk_constraint.initially:
            kw['initially'] = fk_constraint.initially
        if fk_constraint.deferrable:
            kw['deferrable'] = fk_constraint.deferrable
        if fk_constraint.use_alter:
            kw['use_alter'] = fk_constraint.use_alter
        if fk_constraint.match:
            kw['match'] = fk_constraint.match
        source_schema, source_table, source_columns, target_schema, target_table, target_columns, onupdate, ondelete, deferrable, initially = sqla_compat._fk_spec(fk_constraint)
        kw['source_schema'] = source_schema
        kw['referent_schema'] = target_schema
        kw.update(fk_constraint.dialect_kwargs)
        return cls(sqla_compat.constraint_name_or_none(fk_constraint.name), source_table, target_table, source_columns, target_columns, **kw)

    def to_constraint(self, migration_context: Optional[MigrationContext]=None) -> ForeignKeyConstraint:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.foreign_key_constraint(self.constraint_name, self.source_table, self.referent_table, self.local_cols, self.remote_cols, **self.kw)

    @classmethod
    def create_foreign_key(cls, operations: Operations, constraint_name: Optional[str], source_table: str, referent_table: str, local_cols: List[str], remote_cols: List[str], *, onupdate: Optional[str]=None, ondelete: Optional[str]=None, deferrable: Optional[bool]=None, initially: Optional[str]=None, match: Optional[str]=None, source_schema: Optional[str]=None, referent_schema: Optional[str]=None, **dialect_kw: Any) -> None:
        """Issue a "create foreign key" instruction using the
        current migration context.

        e.g.::

            from alembic import op

            op.create_foreign_key(
                "fk_user_address",
                "address",
                "user",
                ["user_id"],
                ["id"],
            )

        This internally generates a :class:`~sqlalchemy.schema.Table` object
        containing the necessary columns, then generates a new
        :class:`~sqlalchemy.schema.ForeignKeyConstraint`
        object which it then associates with the
        :class:`~sqlalchemy.schema.Table`.
        Any event listeners associated with this action will be fired
        off normally.   The :class:`~sqlalchemy.schema.AddConstraint`
        construct is ultimately used to generate the ALTER statement.

        :param constraint_name: Name of the foreign key constraint.  The name
         is necessary so that an ALTER statement can be emitted.  For setups
         that use an automated naming scheme such as that described at
         :ref:`sqla:constraint_naming_conventions`,
         ``name`` here can be ``None``, as the event listener will
         apply the name to the constraint object when it is associated
         with the table.
        :param source_table: String name of the source table.
        :param referent_table: String name of the destination table.
        :param local_cols: a list of string column names in the
         source table.
        :param remote_cols: a list of string column names in the
         remote table.
        :param onupdate: Optional string. If set, emit ON UPDATE <value> when
         issuing DDL for this constraint. Typical values include CASCADE,
         DELETE and RESTRICT.
        :param ondelete: Optional string. If set, emit ON DELETE <value> when
         issuing DDL for this constraint. Typical values include CASCADE,
         DELETE and RESTRICT.
        :param deferrable: optional bool. If set, emit DEFERRABLE or NOT
         DEFERRABLE when issuing DDL for this constraint.
        :param source_schema: Optional schema name of the source table.
        :param referent_schema: Optional schema name of the destination table.

        """
        op = cls(constraint_name, source_table, referent_table, local_cols, remote_cols, onupdate=onupdate, ondelete=ondelete, deferrable=deferrable, source_schema=source_schema, referent_schema=referent_schema, initially=initially, match=match, **dialect_kw)
        return operations.invoke(op)

    @classmethod
    def batch_create_foreign_key(cls, operations: BatchOperations, constraint_name: str, referent_table: str, local_cols: List[str], remote_cols: List[str], *, referent_schema: Optional[str]=None, onupdate: Optional[str]=None, ondelete: Optional[str]=None, deferrable: Optional[bool]=None, initially: Optional[str]=None, match: Optional[str]=None, **dialect_kw: Any) -> None:
        """Issue a "create foreign key" instruction using the
        current batch migration context.

        The batch form of this call omits the ``source`` and ``source_schema``
        arguments from the call.

        e.g.::

            with batch_alter_table("address") as batch_op:
                batch_op.create_foreign_key(
                    "fk_user_address",
                    "user",
                    ["user_id"],
                    ["id"],
                )

        .. seealso::

            :meth:`.Operations.create_foreign_key`

        """
        op = cls(constraint_name, operations.impl.table_name, referent_table, local_cols, remote_cols, onupdate=onupdate, ondelete=ondelete, deferrable=deferrable, source_schema=operations.impl.schema, referent_schema=referent_schema, initially=initially, match=match, **dialect_kw)
        return operations.invoke(op)