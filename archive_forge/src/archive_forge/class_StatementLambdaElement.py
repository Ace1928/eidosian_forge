from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
class StatementLambdaElement(roles.AllowsLambdaRole, LambdaElement, Executable):
    """Represent a composable SQL statement as a :class:`_sql.LambdaElement`.

    The :class:`_sql.StatementLambdaElement` is constructed using the
    :func:`_sql.lambda_stmt` function::


        from sqlalchemy import lambda_stmt

        stmt = lambda_stmt(lambda: select(table))

    Once constructed, additional criteria can be built onto the statement
    by adding subsequent lambdas, which accept the existing statement
    object as a single parameter::

        stmt += lambda s: s.where(table.c.col == parameter)


    .. versionadded:: 1.4

    .. seealso::

        :ref:`engine_lambda_caching`

    """
    if TYPE_CHECKING:

        def __init__(self, fn: _StmtLambdaType, role: Type[SQLRole], opts: Union[Type[LambdaOptions], LambdaOptions]=LambdaOptions, apply_propagate_attrs: Optional[ClauseElement]=None):
            ...

    def __add__(self, other: _StmtLambdaElementType[Any]) -> StatementLambdaElement:
        return self.add_criteria(other)

    def add_criteria(self, other: _StmtLambdaElementType[Any], enable_tracking: bool=True, track_on: Optional[Any]=None, track_closure_variables: bool=True, track_bound_values: bool=True) -> StatementLambdaElement:
        """Add new criteria to this :class:`_sql.StatementLambdaElement`.

        E.g.::

            >>> def my_stmt(parameter):
            ...     stmt = lambda_stmt(
            ...         lambda: select(table.c.x, table.c.y),
            ...     )
            ...     stmt = stmt.add_criteria(
            ...         lambda: table.c.x > parameter
            ...     )
            ...     return stmt

        The :meth:`_sql.StatementLambdaElement.add_criteria` method is
        equivalent to using the Python addition operator to add a new
        lambda, except that additional arguments may be added including
        ``track_closure_values`` and ``track_on``::

            >>> def my_stmt(self, foo):
            ...     stmt = lambda_stmt(
            ...         lambda: select(func.max(foo.x, foo.y)),
            ...         track_closure_variables=False
            ...     )
            ...     stmt = stmt.add_criteria(
            ...         lambda: self.where_criteria,
            ...         track_on=[self]
            ...     )
            ...     return stmt

        See :func:`_sql.lambda_stmt` for a description of the parameters
        accepted.

        """
        opts = self.opts + dict(enable_tracking=enable_tracking, track_closure_variables=track_closure_variables, global_track_bound_values=self.opts.global_track_bound_values, track_on=track_on, track_bound_values=track_bound_values)
        return LinkedLambdaElement(other, parent_lambda=self, opts=opts)

    def _execute_on_connection(self, connection, distilled_params, execution_options):
        if TYPE_CHECKING:
            assert isinstance(self._rec.expected_expr, ClauseElement)
        if self._rec.expected_expr.supports_execution:
            return connection._execute_clauseelement(self, distilled_params, execution_options)
        else:
            raise exc.ObjectNotExecutableError(self)

    @property
    def _proxied(self) -> Any:
        return self._rec_expected_expr

    @property
    def _with_options(self):
        return self._proxied._with_options

    @property
    def _effective_plugin_target(self):
        return self._proxied._effective_plugin_target

    @property
    def _execution_options(self):
        return self._proxied._execution_options

    @property
    def _all_selected_columns(self):
        return self._proxied._all_selected_columns

    @property
    def is_select(self):
        return self._proxied.is_select

    @property
    def is_update(self):
        return self._proxied.is_update

    @property
    def is_insert(self):
        return self._proxied.is_insert

    @property
    def is_text(self):
        return self._proxied.is_text

    @property
    def is_delete(self):
        return self._proxied.is_delete

    @property
    def is_dml(self):
        return self._proxied.is_dml

    def spoil(self) -> NullLambdaStatement:
        """Return a new :class:`.StatementLambdaElement` that will run
        all lambdas unconditionally each time.

        """
        return NullLambdaStatement(self.fn())