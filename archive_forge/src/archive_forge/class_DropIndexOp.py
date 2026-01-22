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
@Operations.register_operation('drop_index')
@BatchOperations.register_operation('drop_index', 'batch_drop_index')
class DropIndexOp(MigrateOperation):
    """Represent a drop index operation."""

    def __init__(self, index_name: Union[quoted_name, str, conv], table_name: Optional[str]=None, *, schema: Optional[str]=None, if_exists: Optional[bool]=None, _reverse: Optional[CreateIndexOp]=None, **kw: Any) -> None:
        self.index_name = index_name
        self.table_name = table_name
        self.schema = schema
        self.if_exists = if_exists
        self._reverse = _reverse
        self.kw = kw

    def to_diff_tuple(self) -> Tuple[str, Index]:
        return ('remove_index', self.to_index())

    def reverse(self) -> CreateIndexOp:
        return CreateIndexOp.from_index(self.to_index())

    @classmethod
    def from_index(cls, index: Index) -> DropIndexOp:
        assert index.table is not None
        return cls(index.name, table_name=index.table.name, schema=index.table.schema, _reverse=CreateIndexOp.from_index(index), unique=index.unique, **index.kwargs)

    def to_index(self, migration_context: Optional[MigrationContext]=None) -> Index:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        return schema_obj.index(self.index_name, self.table_name, self._reverse.columns if self._reverse else ['x'], schema=self.schema, **self.kw)

    @classmethod
    def drop_index(cls, operations: Operations, index_name: str, table_name: Optional[str]=None, *, schema: Optional[str]=None, if_exists: Optional[bool]=None, **kw: Any) -> None:
        """Issue a "drop index" instruction using the current
        migration context.

        e.g.::

            drop_index("accounts")

        :param index_name: name of the index.
        :param table_name: name of the owning table.  Some
         backends such as Microsoft SQL Server require this.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.

        :param if_exists: If True, adds IF EXISTS operator when
         dropping the index.

         .. versionadded:: 1.12.0

        :param \\**kw: Additional keyword arguments not mentioned above are
         dialect specific, and passed in the form
         ``<dialectname>_<argname>``.
         See the documentation regarding an individual dialect at
         :ref:`dialect_toplevel` for detail on documented arguments.

        """
        op = cls(index_name, table_name=table_name, schema=schema, if_exists=if_exists, **kw)
        return operations.invoke(op)

    @classmethod
    def batch_drop_index(cls, operations: BatchOperations, index_name: str, **kw: Any) -> None:
        """Issue a "drop index" instruction using the
        current batch migration context.

        .. seealso::

            :meth:`.Operations.drop_index`

        """
        op = cls(index_name, table_name=operations.impl.table_name, schema=operations.impl.schema, **kw)
        return operations.invoke(op)