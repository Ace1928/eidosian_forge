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
class NullLambdaStatement(roles.AllowsLambdaRole, elements.ClauseElement):
    """Provides the :class:`.StatementLambdaElement` API but does not
    cache or analyze lambdas.

    the lambdas are instead invoked immediately.

    The intended use is to isolate issues that may arise when using
    lambda statements.

    """
    __visit_name__ = 'lambda_element'
    _is_lambda_element = True
    _traverse_internals = [('_resolved', visitors.InternalTraversal.dp_clauseelement)]

    def __init__(self, statement):
        self._resolved = statement
        self._propagate_attrs = statement._propagate_attrs

    def __getattr__(self, key):
        return getattr(self._resolved, key)

    def __add__(self, other):
        statement = other(self._resolved)
        return NullLambdaStatement(statement)

    def add_criteria(self, other, **kw):
        statement = other(self._resolved)
        return NullLambdaStatement(statement)

    def _execute_on_connection(self, connection, distilled_params, execution_options):
        if self._resolved.supports_execution:
            return connection._execute_clauseelement(self, distilled_params, execution_options)
        else:
            raise exc.ObjectNotExecutableError(self)