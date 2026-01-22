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
class TextualSelect(SelectBase, ExecutableReturnsRows, Generative):
    """Wrap a :class:`_expression.TextClause` construct within a
    :class:`_expression.SelectBase`
    interface.

    This allows the :class:`_expression.TextClause` object to gain a
    ``.c`` collection
    and other FROM-like capabilities such as
    :meth:`_expression.FromClause.alias`,
    :meth:`_expression.SelectBase.cte`, etc.

    The :class:`_expression.TextualSelect` construct is produced via the
    :meth:`_expression.TextClause.columns`
    method - see that method for details.

    .. versionchanged:: 1.4 the :class:`_expression.TextualSelect`
       class was renamed
       from ``TextAsFrom``, to more correctly suit its role as a
       SELECT-oriented object and not a FROM clause.

    .. seealso::

        :func:`_expression.text`

        :meth:`_expression.TextClause.columns` - primary creation interface.

    """
    __visit_name__ = 'textual_select'
    _label_style = LABEL_STYLE_NONE
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('column_args', InternalTraversal.dp_clauseelement_list)] + SupportsCloneAnnotations._clone_annotations_traverse_internals
    _is_textual = True
    is_text = True
    is_select = True

    def __init__(self, text: TextClause, columns: List[_ColumnExpressionArgument[Any]], positional: bool=False) -> None:
        self._init(text, [coercions.expect(roles.LabeledColumnExprRole, c) for c in columns], positional)

    def _init(self, text: TextClause, columns: List[NamedColumn[Any]], positional: bool=False) -> None:
        self.element = text
        self.column_args = columns
        self.positional = positional

    @HasMemoized_ro_memoized_attribute
    def selected_columns(self) -> ColumnCollection[str, KeyedColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        this SELECT statement or similar construct returns in its result set,
        not including :class:`_sql.TextClause` constructs.

        This collection differs from the :attr:`_expression.FromClause.columns`
        collection of a :class:`_expression.FromClause` in that the columns
        within this collection cannot be directly nested inside another SELECT
        statement; a subquery must be applied first which provides for the
        necessary parenthesization required by SQL.

        For a :class:`_expression.TextualSelect` construct, the collection
        contains the :class:`_expression.ColumnElement` objects that were
        passed to the constructor, typically via the
        :meth:`_expression.TextClause.columns` method.


        .. versionadded:: 1.4

        """
        return ColumnCollection(((c.key, c) for c in self.column_args)).as_readonly()

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return self.column_args

    def set_label_style(self, style: SelectLabelStyle) -> TextualSelect:
        return self

    def _ensure_disambiguated_names(self) -> TextualSelect:
        return self

    @_generative
    def bindparams(self, *binds: BindParameter[Any], **bind_as_values: Any) -> Self:
        self.element = self.element.bindparams(*binds, **bind_as_values)
        return self

    def _generate_fromclause_column_proxies(self, fromclause: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        if TYPE_CHECKING:
            assert isinstance(fromclause, Subquery)
        if proxy_compound_columns:
            fromclause._columns._populate_separate_keys((c._make_proxy(fromclause, compound_select_cols=extra_cols) for c, extra_cols in zip(self.column_args, proxy_compound_columns)))
        else:
            fromclause._columns._populate_separate_keys((c._make_proxy(fromclause) for c in self.column_args))

    def _scalar_type(self) -> Union[TypeEngine[Any], Any]:
        return self.column_args[0].type