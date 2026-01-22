from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class CompilerElement(Visitable):
    """base class for SQL elements that can be compiled to produce a
    SQL string.

    .. versionadded:: 2.0

    """
    __slots__ = ()
    __visit_name__ = 'compiler_element'
    supports_execution = False
    stringify_dialect = 'default'

    @util.preload_module('sqlalchemy.engine.default')
    @util.preload_module('sqlalchemy.engine.url')
    def compile(self, bind: Optional[_HasDialect]=None, dialect: Optional[Dialect]=None, **kw: Any) -> Compiled:
        """Compile this SQL expression.

        The return value is a :class:`~.Compiled` object.
        Calling ``str()`` or ``unicode()`` on the returned value will yield a
        string representation of the result. The
        :class:`~.Compiled` object also can return a
        dictionary of bind parameter names and values
        using the ``params`` accessor.

        :param bind: An :class:`.Connection` or :class:`.Engine` which
           can provide a :class:`.Dialect` in order to generate a
           :class:`.Compiled` object.  If the ``bind`` and
           ``dialect`` parameters are both omitted, a default SQL compiler
           is used.

        :param column_keys: Used for INSERT and UPDATE statements, a list of
            column names which should be present in the VALUES clause of the
            compiled statement. If ``None``, all columns from the target table
            object are rendered.

        :param dialect: A :class:`.Dialect` instance which can generate
            a :class:`.Compiled` object.  This argument takes precedence over
            the ``bind`` argument.

        :param compile_kwargs: optional dictionary of additional parameters
            that will be passed through to the compiler within all "visit"
            methods.  This allows any custom flag to be passed through to
            a custom compilation construct, for example.  It is also used
            for the case of passing the ``literal_binds`` flag through::

                from sqlalchemy.sql import table, column, select

                t = table('t', column('x'))

                s = select(t).where(t.c.x == 5)

                print(s.compile(compile_kwargs={"literal_binds": True}))

        .. seealso::

            :ref:`faq_sql_expression_string`

        """
        if dialect is None:
            if bind:
                dialect = bind.dialect
            elif self.stringify_dialect == 'default':
                default = util.preloaded.engine_default
                dialect = default.StrCompileDialect()
            else:
                url = util.preloaded.engine_url
                dialect = url.URL.create(self.stringify_dialect).get_dialect()()
        return self._compiler(dialect, **kw)

    def _compiler(self, dialect: Dialect, **kw: Any) -> Compiled:
        """Return a compiler appropriate for this ClauseElement, given a
        Dialect."""
        if TYPE_CHECKING:
            assert isinstance(self, ClauseElement)
        return dialect.statement_compiler(dialect, self, **kw)

    def __str__(self) -> str:
        return str(self.compile())