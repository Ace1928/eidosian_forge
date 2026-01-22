from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
class SelectLabelStyle(Enum):
    """Label style constants that may be passed to
    :meth:`_sql.Select.set_label_style`."""
    LABEL_STYLE_NONE = 0
    'Label style indicating no automatic labeling should be applied to the\n    columns clause of a SELECT statement.\n\n    Below, the columns named ``columna`` are both rendered as is, meaning that\n    the name ``columna`` can only refer to the first occurrence of this name\n    within a result set, as well as if the statement were used as a subquery:\n\n    .. sourcecode:: pycon+sql\n\n        >>> from sqlalchemy import table, column, select, true, LABEL_STYLE_NONE\n        >>> table1 = table("table1", column("columna"), column("columnb"))\n        >>> table2 = table("table2", column("columna"), column("columnc"))\n        >>> print(select(table1, table2).join(table2, true()).set_label_style(LABEL_STYLE_NONE))\n        {printsql}SELECT table1.columna, table1.columnb, table2.columna, table2.columnc\n        FROM table1 JOIN table2 ON true\n\n    Used with the :meth:`_sql.Select.set_label_style` method.\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_TABLENAME_PLUS_COL = 1
    'Label style indicating all columns should be labeled as\n    ``<tablename>_<columnname>`` when generating the columns clause of a SELECT\n    statement, to disambiguate same-named columns referenced from different\n    tables, aliases, or subqueries.\n\n    Below, all column names are given a label so that the two same-named\n    columns ``columna`` are disambiguated as ``table1_columna`` and\n    ``table2_columna``:\n\n    .. sourcecode:: pycon+sql\n\n        >>> from sqlalchemy import table, column, select, true, LABEL_STYLE_TABLENAME_PLUS_COL\n        >>> table1 = table("table1", column("columna"), column("columnb"))\n        >>> table2 = table("table2", column("columna"), column("columnc"))\n        >>> print(select(table1, table2).join(table2, true()).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL))\n        {printsql}SELECT table1.columna AS table1_columna, table1.columnb AS table1_columnb, table2.columna AS table2_columna, table2.columnc AS table2_columnc\n        FROM table1 JOIN table2 ON true\n\n    Used with the :meth:`_sql.GenerativeSelect.set_label_style` method.\n    Equivalent to the legacy method ``Select.apply_labels()``;\n    :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL` is SQLAlchemy\'s legacy\n    auto-labeling style. :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` provides a\n    less intrusive approach to disambiguation of same-named column expressions.\n\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_DISAMBIGUATE_ONLY = 2
    'Label style indicating that columns with a name that conflicts with\n    an existing name should be labeled with a semi-anonymizing label\n    when generating the columns clause of a SELECT statement.\n\n    Below, most column names are left unaffected, except for the second\n    occurrence of the name ``columna``, which is labeled using the\n    label ``columna_1`` to disambiguate it from that of ``tablea.columna``:\n\n    .. sourcecode:: pycon+sql\n\n        >>> from sqlalchemy import table, column, select, true, LABEL_STYLE_DISAMBIGUATE_ONLY\n        >>> table1 = table("table1", column("columna"), column("columnb"))\n        >>> table2 = table("table2", column("columna"), column("columnc"))\n        >>> print(select(table1, table2).join(table2, true()).set_label_style(LABEL_STYLE_DISAMBIGUATE_ONLY))\n        {printsql}SELECT table1.columna, table1.columnb, table2.columna AS columna_1, table2.columnc\n        FROM table1 JOIN table2 ON true\n\n    Used with the :meth:`_sql.GenerativeSelect.set_label_style` method,\n    :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` is the default labeling style\n    for all SELECT statements outside of :term:`1.x style` ORM queries.\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_DEFAULT = LABEL_STYLE_DISAMBIGUATE_ONLY
    'The default label style, refers to\n    :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY`.\n\n    .. versionadded:: 1.4\n\n    '
    LABEL_STYLE_LEGACY_ORM = 3