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
class ForeignKeyConstraint(ColumnCollectionConstraint):
    """A table-level FOREIGN KEY constraint.

    Defines a single column or composite FOREIGN KEY ... REFERENCES
    constraint. For a no-frills, single column foreign key, adding a
    :class:`_schema.ForeignKey` to the definition of a :class:`_schema.Column`
    is a
    shorthand equivalent for an unnamed, single column
    :class:`_schema.ForeignKeyConstraint`.

    Examples of foreign key configuration are in :ref:`metadata_foreignkeys`.

    """
    __visit_name__ = 'foreign_key_constraint'

    def __init__(self, columns: _typing_Sequence[_DDLColumnArgument], refcolumns: _typing_Sequence[_DDLColumnArgument], name: _ConstraintNameArgument=None, onupdate: Optional[str]=None, ondelete: Optional[str]=None, deferrable: Optional[bool]=None, initially: Optional[str]=None, use_alter: bool=False, link_to_name: bool=False, match: Optional[str]=None, table: Optional[Table]=None, info: Optional[_InfoType]=None, comment: Optional[str]=None, **dialect_kw: Any) -> None:
        """Construct a composite-capable FOREIGN KEY.

        :param columns: A sequence of local column names. The named columns
          must be defined and present in the parent Table. The names should
          match the ``key`` given to each column (defaults to the name) unless
          ``link_to_name`` is True.

        :param refcolumns: A sequence of foreign column names or Column
          objects. The columns must all be located within the same Table.

        :param name: Optional, the in-database name of the key.

        :param onupdate: Optional string. If set, emit ON UPDATE <value> when
          issuing DDL for this constraint. Typical values include CASCADE,
          DELETE and RESTRICT.

        :param ondelete: Optional string. If set, emit ON DELETE <value> when
          issuing DDL for this constraint. Typical values include CASCADE,
          DELETE and RESTRICT.

        :param deferrable: Optional bool. If set, emit DEFERRABLE or NOT
          DEFERRABLE when issuing DDL for this constraint.

        :param initially: Optional string. If set, emit INITIALLY <value> when
          issuing DDL for this constraint.

        :param link_to_name: if True, the string name given in ``column`` is
          the rendered name of the referenced column, not its locally assigned
          ``key``.

        :param use_alter: If True, do not emit the DDL for this constraint as
          part of the CREATE TABLE definition. Instead, generate it via an
          ALTER TABLE statement issued after the full collection of tables
          have been created, and drop it via an ALTER TABLE statement before
          the full collection of tables are dropped.

          The use of :paramref:`_schema.ForeignKeyConstraint.use_alter` is
          particularly geared towards the case where two or more tables
          are established within a mutually-dependent foreign key constraint
          relationship; however, the :meth:`_schema.MetaData.create_all` and
          :meth:`_schema.MetaData.drop_all`
          methods will perform this resolution
          automatically, so the flag is normally not needed.

          .. seealso::

                :ref:`use_alter`

        :param match: Optional string. If set, emit MATCH <value> when issuing
          DDL for this constraint. Typical values include SIMPLE, PARTIAL
          and FULL.

        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        :param comment: Optional string that will render an SQL comment on
          foreign key constraint creation.

            .. versionadded:: 2.0

        :param \\**dialect_kw:  Additional keyword arguments are dialect
          specific, and passed in the form ``<dialectname>_<argname>``.  See
          the documentation regarding an individual dialect at
          :ref:`dialect_toplevel` for detail on documented arguments.

        """
        Constraint.__init__(self, name=name, deferrable=deferrable, initially=initially, info=info, comment=comment, **dialect_kw)
        self.onupdate = onupdate
        self.ondelete = ondelete
        self.link_to_name = link_to_name
        self.use_alter = use_alter
        self.match = match
        if len(set(columns)) != len(refcolumns):
            if len(set(columns)) != len(columns):
                raise exc.ArgumentError('ForeignKeyConstraint with duplicate source column references are not supported.')
            else:
                raise exc.ArgumentError('ForeignKeyConstraint number of constrained columns must match the number of referenced columns.')
        self.elements = [ForeignKey(refcol, _constraint=self, name=self.name, onupdate=self.onupdate, ondelete=self.ondelete, use_alter=self.use_alter, link_to_name=self.link_to_name, match=self.match, deferrable=self.deferrable, initially=self.initially, **self.dialect_kwargs) for refcol in refcolumns]
        ColumnCollectionMixin.__init__(self, *columns)
        if table is not None:
            if hasattr(self, 'parent'):
                assert table is self.parent
            self._set_parent_with_dispatch(table)

    def _append_element(self, column: Column[Any], fk: ForeignKey) -> None:
        self._columns.add(column)
        self.elements.append(fk)
    columns: ReadOnlyColumnCollection[str, Column[Any]]
    'A :class:`_expression.ColumnCollection` representing the set of columns\n    for this constraint.\n\n    '
    elements: List[ForeignKey]
    'A sequence of :class:`_schema.ForeignKey` objects.\n\n    Each :class:`_schema.ForeignKey`\n    represents a single referring column/referred\n    column pair.\n\n    This collection is intended to be read-only.\n\n    '

    @property
    def _elements(self) -> util.OrderedDict[str, ForeignKey]:
        return util.OrderedDict(zip(self.column_keys, self.elements))

    @property
    def _referred_schema(self) -> Optional[str]:
        for elem in self.elements:
            return elem._referred_schema
        else:
            return None

    @property
    def referred_table(self) -> Table:
        """The :class:`_schema.Table` object to which this
        :class:`_schema.ForeignKeyConstraint` references.

        This is a dynamically calculated attribute which may not be available
        if the constraint and/or parent table is not yet associated with
        a metadata collection that contains the referred table.

        """
        return self.elements[0].column.table

    def _validate_dest_table(self, table: Table) -> None:
        table_keys = {elem._table_key() for elem in self.elements}
        if None not in table_keys and len(table_keys) > 1:
            elem0, elem1 = sorted(table_keys)[0:2]
            raise exc.ArgumentError(f'ForeignKeyConstraint on {table.fullname}({self._col_description}) refers to multiple remote tables: {elem0} and {elem1}')

    @property
    def column_keys(self) -> _typing_Sequence[str]:
        """Return a list of string keys representing the local
        columns in this :class:`_schema.ForeignKeyConstraint`.

        This list is either the original string arguments sent
        to the constructor of the :class:`_schema.ForeignKeyConstraint`,
        or if the constraint has been initialized with :class:`_schema.Column`
        objects, is the string ``.key`` of each element.

        """
        if hasattr(self, 'parent'):
            return self._columns.keys()
        else:
            return [col.key if isinstance(col, ColumnElement) else str(col) for col in self._pending_colargs]

    @property
    def _col_description(self) -> str:
        return ', '.join(self.column_keys)

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        table = parent
        assert isinstance(table, Table)
        Constraint._set_parent(self, table)
        ColumnCollectionConstraint._set_parent(self, table)
        for col, fk in zip(self._columns, self.elements):
            if not hasattr(fk, 'parent') or fk.parent is not col:
                fk._set_parent_with_dispatch(col)
        self._validate_dest_table(table)

    @util.deprecated('1.4', 'The :meth:`_schema.ForeignKeyConstraint.copy` method is deprecated and will be removed in a future release.')
    def copy(self, *, schema: Optional[str]=None, target_table: Optional[Table]=None, **kw: Any) -> ForeignKeyConstraint:
        return self._copy(schema=schema, target_table=target_table, **kw)

    def _copy(self, *, schema: Optional[str]=None, target_table: Optional[Table]=None, **kw: Any) -> ForeignKeyConstraint:
        fkc = ForeignKeyConstraint([x.parent.key for x in self.elements], [x._get_colspec(schema=schema, table_name=target_table.name if target_table is not None and x._table_key() == x.parent.table.key else None, _is_copy=True) for x in self.elements], name=self.name, onupdate=self.onupdate, ondelete=self.ondelete, use_alter=self.use_alter, deferrable=self.deferrable, initially=self.initially, link_to_name=self.link_to_name, match=self.match, comment=self.comment)
        for self_fk, other_fk in zip(self.elements, fkc.elements):
            self_fk._schema_item_copy(other_fk)
        return self._schema_item_copy(fkc)