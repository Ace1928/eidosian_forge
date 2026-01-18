from __future__ import annotations
from collections import deque
import copy
from itertools import chain
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import visitors
from ._typing import is_text_clause
from .annotation import _deep_annotate as _deep_annotate  # noqa: F401
from .annotation import _deep_deannotate as _deep_deannotate  # noqa: F401
from .annotation import _shallow_annotate as _shallow_annotate  # noqa: F401
from .base import _expand_cloned
from .base import _from_objects
from .cache_key import HasCacheKey as HasCacheKey  # noqa: F401
from .ddl import sort_tables as sort_tables  # noqa: F401
from .elements import _find_columns as _find_columns
from .elements import _label_reference
from .elements import _textual_label_reference
from .elements import BindParameter
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Grouping
from .elements import KeyedColumnElement
from .elements import Label
from .elements import NamedColumn
from .elements import Null
from .elements import UnaryExpression
from .schema import Column
from .selectable import Alias
from .selectable import FromClause
from .selectable import FromGrouping
from .selectable import Join
from .selectable import ScalarSelect
from .selectable import SelectBase
from .selectable import TableClause
from .visitors import _ET
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def visit_binary_product(fn: Callable[[BinaryExpression[Any], ColumnElement[Any], ColumnElement[Any]], None], expr: ColumnElement[Any]) -> None:
    """Produce a traversal of the given expression, delivering
    column comparisons to the given function.

    The function is of the form::

        def my_fn(binary, left, right)

    For each binary expression located which has a
    comparison operator, the product of "left" and
    "right" will be delivered to that function,
    in terms of that binary.

    Hence an expression like::

        and_(
            (a + b) == q + func.sum(e + f),
            j == r
        )

    would have the traversal::

        a <eq> q
        a <eq> e
        a <eq> f
        b <eq> q
        b <eq> e
        b <eq> f
        j <eq> r

    That is, every combination of "left" and
    "right" that doesn't further contain
    a binary comparison is passed as pairs.

    """
    stack: List[BinaryExpression[Any]] = []

    def visit(element: ClauseElement) -> Iterator[ColumnElement[Any]]:
        if isinstance(element, ScalarSelect):
            yield element
        elif element.__visit_name__ == 'binary' and operators.is_comparison(element.operator):
            stack.insert(0, element)
            for l in visit(element.left):
                for r in visit(element.right):
                    fn(stack[0], l, r)
            stack.pop(0)
            for elem in element.get_children():
                visit(elem)
        else:
            if isinstance(element, ColumnClause):
                yield element
            for elem in element.get_children():
                yield from visit(elem)
    list(visit(expr))
    visit = None