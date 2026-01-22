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
@Operations.register_operation('create_table_comment')
@BatchOperations.register_operation('create_table_comment', 'batch_create_table_comment')
class CreateTableCommentOp(AlterTableOp):
    """Represent a COMMENT ON `table` operation."""

    def __init__(self, table_name: str, comment: Optional[str], *, schema: Optional[str]=None, existing_comment: Optional[str]=None) -> None:
        self.table_name = table_name
        self.comment = comment
        self.existing_comment = existing_comment
        self.schema = schema

    @classmethod
    def create_table_comment(cls, operations: Operations, table_name: str, comment: Optional[str], *, existing_comment: Optional[str]=None, schema: Optional[str]=None) -> None:
        """Emit a COMMENT ON operation to set the comment for a table.

        :param table_name: string name of the target table.
        :param comment: string value of the comment being registered against
         the specified table.
        :param existing_comment: String value of a comment
         already registered on the specified table, used within autogenerate
         so that the operation is reversible, but not required for direct
         use.

        .. seealso::

            :meth:`.Operations.drop_table_comment`

            :paramref:`.Operations.alter_column.comment`

        """
        op = cls(table_name, comment, existing_comment=existing_comment, schema=schema)
        return operations.invoke(op)

    @classmethod
    def batch_create_table_comment(cls, operations: BatchOperations, comment: Optional[str], *, existing_comment: Optional[str]=None) -> None:
        """Emit a COMMENT ON operation to set the comment for a table
        using the current batch migration context.

        :param comment: string value of the comment being registered against
         the specified table.
        :param existing_comment: String value of a comment
         already registered on the specified table, used within autogenerate
         so that the operation is reversible, but not required for direct
         use.

        """
        op = cls(operations.impl.table_name, comment, existing_comment=existing_comment, schema=operations.impl.schema)
        return operations.invoke(op)

    def reverse(self) -> Union[CreateTableCommentOp, DropTableCommentOp]:
        """Reverses the COMMENT ON operation against a table."""
        if self.existing_comment is None:
            return DropTableCommentOp(self.table_name, existing_comment=self.comment, schema=self.schema)
        else:
            return CreateTableCommentOp(self.table_name, self.existing_comment, existing_comment=self.comment, schema=self.schema)

    def to_table(self, migration_context: Optional[MigrationContext]=None) -> Table:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.table(self.table_name, schema=self.schema, comment=self.comment)

    def to_diff_tuple(self) -> Tuple[Any, ...]:
        return ('add_table_comment', self.to_table(), self.existing_comment)