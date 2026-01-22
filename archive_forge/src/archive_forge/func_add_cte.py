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
@_generative
def add_cte(self, *ctes: CTE, nest_here: bool=False) -> Self:
    """Add one or more :class:`_sql.CTE` constructs to this statement.

        This method will associate the given :class:`_sql.CTE` constructs with
        the parent statement such that they will each be unconditionally
        rendered in the WITH clause of the final statement, even if not
        referenced elsewhere within the statement or any sub-selects.

        The optional :paramref:`.HasCTE.add_cte.nest_here` parameter when set
        to True will have the effect that each given :class:`_sql.CTE` will
        render in a WITH clause rendered directly along with this statement,
        rather than being moved to the top of the ultimate rendered statement,
        even if this statement is rendered as a subquery within a larger
        statement.

        This method has two general uses. One is to embed CTE statements that
        serve some purpose without being referenced explicitly, such as the use
        case of embedding a DML statement such as an INSERT or UPDATE as a CTE
        inline with a primary statement that may draw from its results
        indirectly.  The other is to provide control over the exact placement
        of a particular series of CTE constructs that should remain rendered
        directly in terms of a particular statement that may be nested in a
        larger statement.

        E.g.::

            from sqlalchemy import table, column, select
            t = table('t', column('c1'), column('c2'))

            ins = t.insert().values({"c1": "x", "c2": "y"}).cte()

            stmt = select(t).add_cte(ins)

        Would render::

            WITH anon_1 AS
            (INSERT INTO t (c1, c2) VALUES (:param_1, :param_2))
            SELECT t.c1, t.c2
            FROM t

        Above, the "anon_1" CTE is not referenced in the SELECT
        statement, however still accomplishes the task of running an INSERT
        statement.

        Similarly in a DML-related context, using the PostgreSQL
        :class:`_postgresql.Insert` construct to generate an "upsert"::

            from sqlalchemy import table, column
            from sqlalchemy.dialects.postgresql import insert

            t = table("t", column("c1"), column("c2"))

            delete_statement_cte = (
                t.delete().where(t.c.c1 < 1).cte("deletions")
            )

            insert_stmt = insert(t).values({"c1": 1, "c2": 2})
            update_statement = insert_stmt.on_conflict_do_update(
                index_elements=[t.c.c1],
                set_={
                    "c1": insert_stmt.excluded.c1,
                    "c2": insert_stmt.excluded.c2,
                },
            ).add_cte(delete_statement_cte)

            print(update_statement)

        The above statement renders as::

            WITH deletions AS
            (DELETE FROM t WHERE t.c1 < %(c1_1)s)
            INSERT INTO t (c1, c2) VALUES (%(c1)s, %(c2)s)
            ON CONFLICT (c1) DO UPDATE SET c1 = excluded.c1, c2 = excluded.c2

        .. versionadded:: 1.4.21

        :param \\*ctes: zero or more :class:`.CTE` constructs.

         .. versionchanged:: 2.0  Multiple CTE instances are accepted

        :param nest_here: if True, the given CTE or CTEs will be rendered
         as though they specified the :paramref:`.HasCTE.cte.nesting` flag
         to ``True`` when they were added to this :class:`.HasCTE`.
         Assuming the given CTEs are not referenced in an outer-enclosing
         statement as well, the CTEs given should render at the level of
         this statement when this flag is given.

         .. versionadded:: 2.0

         .. seealso::

            :paramref:`.HasCTE.cte.nesting`


        """
    opt = _CTEOpts(nest_here)
    for cte in ctes:
        cte = coercions.expect(roles.IsCTERole, cte)
        self._independent_ctes += (cte,)
        self._independent_ctes_opts += (opt,)
    return self