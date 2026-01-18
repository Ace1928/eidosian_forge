from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
def to_metadata(self, metadata: MetaData, schema: Union[str, Literal[SchemaConst.RETAIN_SCHEMA]]=RETAIN_SCHEMA, referred_schema_fn: Optional[Callable[[Table, Optional[str], ForeignKeyConstraint, Optional[str]], Optional[str]]]=None, name: Optional[str]=None) -> Table:
    """Return a copy of this :class:`_schema.Table` associated with a
        different :class:`_schema.MetaData`.

        E.g.::

            m1 = MetaData()

            user = Table('user', m1, Column('id', Integer, primary_key=True))

            m2 = MetaData()
            user_copy = user.to_metadata(m2)

        .. versionchanged:: 1.4  The :meth:`_schema.Table.to_metadata` function
           was renamed from :meth:`_schema.Table.tometadata`.


        :param metadata: Target :class:`_schema.MetaData` object,
         into which the
         new :class:`_schema.Table` object will be created.

        :param schema: optional string name indicating the target schema.
         Defaults to the special symbol :attr:`.RETAIN_SCHEMA` which indicates
         that no change to the schema name should be made in the new
         :class:`_schema.Table`.  If set to a string name, the new
         :class:`_schema.Table`
         will have this new name as the ``.schema``.  If set to ``None``, the
         schema will be set to that of the schema set on the target
         :class:`_schema.MetaData`, which is typically ``None`` as well,
         unless
         set explicitly::

            m2 = MetaData(schema='newschema')

            # user_copy_one will have "newschema" as the schema name
            user_copy_one = user.to_metadata(m2, schema=None)

            m3 = MetaData()  # schema defaults to None

            # user_copy_two will have None as the schema name
            user_copy_two = user.to_metadata(m3, schema=None)

        :param referred_schema_fn: optional callable which can be supplied
         in order to provide for the schema name that should be assigned
         to the referenced table of a :class:`_schema.ForeignKeyConstraint`.
         The callable accepts this parent :class:`_schema.Table`, the
         target schema that we are changing to, the
         :class:`_schema.ForeignKeyConstraint` object, and the existing
         "target schema" of that constraint.  The function should return the
         string schema name that should be applied.    To reset the schema
         to "none", return the symbol :data:`.BLANK_SCHEMA`.  To effect no
         change, return ``None`` or :data:`.RETAIN_SCHEMA`.

         .. versionchanged:: 1.4.33  The ``referred_schema_fn`` function
            may return the :data:`.BLANK_SCHEMA` or :data:`.RETAIN_SCHEMA`
            symbols.

         E.g.::

                def referred_schema_fn(table, to_schema,
                                                constraint, referred_schema):
                    if referred_schema == 'base_tables':
                        return referred_schema
                    else:
                        return to_schema

                new_table = table.to_metadata(m2, schema="alt_schema",
                                        referred_schema_fn=referred_schema_fn)

        :param name: optional string name indicating the target table name.
         If not specified or None, the table name is retained.  This allows
         a :class:`_schema.Table` to be copied to the same
         :class:`_schema.MetaData` target
         with a new name.

        """
    if name is None:
        name = self.name
    actual_schema: Optional[str]
    if schema is RETAIN_SCHEMA:
        actual_schema = self.schema
    elif schema is None:
        actual_schema = metadata.schema
    else:
        actual_schema = schema
    key = _get_table_key(name, actual_schema)
    if key in metadata.tables:
        util.warn(f"Table '{self.description}' already exists within the given MetaData - not copying.")
        return metadata.tables[key]
    args = []
    for col in self.columns:
        args.append(col._copy(schema=actual_schema))
    table = Table(name, metadata, *args, schema=actual_schema, comment=self.comment, **self.kwargs)
    for const in self.constraints:
        if isinstance(const, ForeignKeyConstraint):
            referred_schema = const._referred_schema
            if referred_schema_fn:
                fk_constraint_schema = referred_schema_fn(self, actual_schema, const, referred_schema)
            else:
                fk_constraint_schema = actual_schema if referred_schema == self.schema else None
            table.append_constraint(const._copy(schema=fk_constraint_schema, target_table=table))
        elif not const._type_bound:
            if const._column_flag:
                continue
            table.append_constraint(const._copy(schema=actual_schema, target_table=table))
    for index in self.indexes:
        if index._column_flag:
            continue
        Index(index.name, *[_copy_expression(expr, self, table) for expr in index._table_bound_expressions], unique=index.unique, _table=table, **index.kwargs)
    return self._schema_item_copy(table)