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
@Operations.register_operation('create_unique_constraint')
@BatchOperations.register_operation('create_unique_constraint', 'batch_create_unique_constraint')
@AddConstraintOp.register_add_constraint('unique_constraint')
class CreateUniqueConstraintOp(AddConstraintOp):
    """Represent a create unique constraint operation."""
    constraint_type = 'unique'

    def __init__(self, constraint_name: Optional[sqla_compat._ConstraintNameDefined], table_name: str, columns: Sequence[str], *, schema: Optional[str]=None, **kw: Any) -> None:
        self.constraint_name = constraint_name
        self.table_name = table_name
        self.columns = columns
        self.schema = schema
        self.kw = kw

    @classmethod
    def from_constraint(cls, constraint: Constraint) -> CreateUniqueConstraintOp:
        constraint_table = sqla_compat._table_for_constraint(constraint)
        uq_constraint = cast('UniqueConstraint', constraint)
        kw: Dict[str, Any] = {}
        if uq_constraint.deferrable:
            kw['deferrable'] = uq_constraint.deferrable
        if uq_constraint.initially:
            kw['initially'] = uq_constraint.initially
        kw.update(uq_constraint.dialect_kwargs)
        return cls(sqla_compat.constraint_name_or_none(uq_constraint.name), constraint_table.name, [c.name for c in uq_constraint.columns], schema=constraint_table.schema, **kw)

    def to_constraint(self, migration_context: Optional[MigrationContext]=None) -> UniqueConstraint:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.unique_constraint(self.constraint_name, self.table_name, self.columns, schema=self.schema, **self.kw)

    @classmethod
    def create_unique_constraint(cls, operations: Operations, constraint_name: Optional[str], table_name: str, columns: Sequence[str], *, schema: Optional[str]=None, **kw: Any) -> Any:
        """Issue a "create unique constraint" instruction using the
        current migration context.

        e.g.::

            from alembic import op
            op.create_unique_constraint("uq_user_name", "user", ["name"])

        This internally generates a :class:`~sqlalchemy.schema.Table` object
        containing the necessary columns, then generates a new
        :class:`~sqlalchemy.schema.UniqueConstraint`
        object which it then associates with the
        :class:`~sqlalchemy.schema.Table`.
        Any event listeners associated with this action will be fired
        off normally.   The :class:`~sqlalchemy.schema.AddConstraint`
        construct is ultimately used to generate the ALTER statement.

        :param name: Name of the unique constraint.  The name is necessary
         so that an ALTER statement can be emitted.  For setups that
         use an automated naming scheme such as that described at
         :ref:`sqla:constraint_naming_conventions`,
         ``name`` here can be ``None``, as the event listener will
         apply the name to the constraint object when it is associated
         with the table.
        :param table_name: String name of the source table.
        :param columns: a list of string column names in the
         source table.
        :param deferrable: optional bool. If set, emit DEFERRABLE or
         NOT DEFERRABLE when issuing DDL for this constraint.
        :param initially: optional string. If set, emit INITIALLY <value>
         when issuing DDL for this constraint.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.

        """
        op = cls(constraint_name, table_name, columns, schema=schema, **kw)
        return operations.invoke(op)

    @classmethod
    def batch_create_unique_constraint(cls, operations: BatchOperations, constraint_name: str, columns: Sequence[str], **kw: Any) -> Any:
        """Issue a "create unique constraint" instruction using the
        current batch migration context.

        The batch form of this call omits the ``source`` and ``schema``
        arguments from the call.

        .. seealso::

            :meth:`.Operations.create_unique_constraint`

        """
        kw['schema'] = operations.impl.schema
        op = cls(constraint_name, operations.impl.table_name, columns, **kw)
        return operations.invoke(op)