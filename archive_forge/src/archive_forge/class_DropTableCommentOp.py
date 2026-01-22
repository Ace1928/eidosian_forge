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
@Operations.register_operation('drop_table_comment')
@BatchOperations.register_operation('drop_table_comment', 'batch_drop_table_comment')
class DropTableCommentOp(AlterTableOp):
    """Represent an operation to remove the comment from a table."""

    def __init__(self, table_name: str, *, schema: Optional[str]=None, existing_comment: Optional[str]=None) -> None:
        self.table_name = table_name
        self.existing_comment = existing_comment
        self.schema = schema

    @classmethod
    def drop_table_comment(cls, operations: Operations, table_name: str, *, existing_comment: Optional[str]=None, schema: Optional[str]=None) -> None:
        """Issue a "drop table comment" operation to
        remove an existing comment set on a table.

        :param table_name: string name of the target table.
        :param existing_comment: An optional string value of a comment already
         registered on the specified table.

        .. seealso::

            :meth:`.Operations.create_table_comment`

            :paramref:`.Operations.alter_column.comment`

        """
        op = cls(table_name, existing_comment=existing_comment, schema=schema)
        return operations.invoke(op)

    @classmethod
    def batch_drop_table_comment(cls, operations: BatchOperations, *, existing_comment: Optional[str]=None) -> None:
        """Issue a "drop table comment" operation to
        remove an existing comment set on a table using the current
        batch operations context.

        :param existing_comment: An optional string value of a comment already
         registered on the specified table.

        """
        op = cls(operations.impl.table_name, existing_comment=existing_comment, schema=operations.impl.schema)
        return operations.invoke(op)

    def reverse(self) -> CreateTableCommentOp:
        """Reverses the COMMENT ON operation against a table."""
        return CreateTableCommentOp(self.table_name, self.existing_comment, schema=self.schema)

    def to_table(self, migration_context: Optional[MigrationContext]=None) -> Table:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.table(self.table_name, schema=self.schema)

    def to_diff_tuple(self) -> Tuple[Any, ...]:
        return ('remove_table_comment', self.to_table())