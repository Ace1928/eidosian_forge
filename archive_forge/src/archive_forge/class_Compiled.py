from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
class Compiled:
    """Represent a compiled SQL or DDL expression.

    The ``__str__`` method of the ``Compiled`` object should produce
    the actual text of the statement.  ``Compiled`` objects are
    specific to their underlying database dialect, and also may
    or may not be specific to the columns referenced within a
    particular set of bind parameters.  In no case should the
    ``Compiled`` object be dependent on the actual values of those
    bind parameters, even though it may reference those values as
    defaults.
    """
    statement: Optional[ClauseElement] = None
    'The statement to compile.'
    string: str = ''
    'The string representation of the ``statement``'
    state: CompilerState
    "description of the compiler's state"
    is_sql = False
    is_ddl = False
    _cached_metadata: Optional[CursorResultMetaData] = None
    _result_columns: Optional[List[ResultColumnsEntry]] = None
    schema_translate_map: Optional[SchemaTranslateMapType] = None
    execution_options: _ExecuteOptions = util.EMPTY_DICT
    '\n    Execution options propagated from the statement.   In some cases,\n    sub-elements of the statement can modify these.\n    '
    preparer: IdentifierPreparer
    _annotations: _AnnotationDict = util.EMPTY_DICT
    compile_state: Optional[CompileState] = None
    'Optional :class:`.CompileState` object that maintains additional\n    state used by the compiler.\n\n    Major executable objects such as :class:`_expression.Insert`,\n    :class:`_expression.Update`, :class:`_expression.Delete`,\n    :class:`_expression.Select` will generate this\n    state when compiled in order to calculate additional information about the\n    object.   For the top level object that is to be executed, the state can be\n    stored here where it can also have applicability towards result set\n    processing.\n\n    .. versionadded:: 1.4\n\n    '
    dml_compile_state: Optional[CompileState] = None
    'Optional :class:`.CompileState` assigned at the same point that\n    .isinsert, .isupdate, or .isdelete is assigned.\n\n    This will normally be the same object as .compile_state, with the\n    exception of cases like the :class:`.ORMFromStatementCompileState`\n    object.\n\n    .. versionadded:: 1.4.40\n\n    '
    cache_key: Optional[CacheKey] = None
    "The :class:`.CacheKey` that was generated ahead of creating this\n    :class:`.Compiled` object.\n\n    This is used for routines that need access to the original\n    :class:`.CacheKey` instance generated when the :class:`.Compiled`\n    instance was first cached, typically in order to reconcile\n    the original list of :class:`.BindParameter` objects with a\n    per-statement list that's generated on each call.\n\n    "
    _gen_time: float
    'Generation time of this :class:`.Compiled`, used for reporting\n    cache stats.'

    def __init__(self, dialect: Dialect, statement: Optional[ClauseElement], schema_translate_map: Optional[SchemaTranslateMapType]=None, render_schema_translate: bool=False, compile_kwargs: Mapping[str, Any]=util.immutabledict()):
        """Construct a new :class:`.Compiled` object.

        :param dialect: :class:`.Dialect` to compile against.

        :param statement: :class:`_expression.ClauseElement` to be compiled.

        :param schema_translate_map: dictionary of schema names to be
         translated when forming the resultant SQL

         .. seealso::

            :ref:`schema_translating`

        :param compile_kwargs: additional kwargs that will be
         passed to the initial call to :meth:`.Compiled.process`.


        """
        self.dialect = dialect
        self.preparer = self.dialect.identifier_preparer
        if schema_translate_map:
            self.schema_translate_map = schema_translate_map
            self.preparer = self.preparer._with_schema_translate(schema_translate_map)
        if statement is not None:
            self.state = CompilerState.COMPILING
            self.statement = statement
            self.can_execute = statement.supports_execution
            self._annotations = statement._annotations
            if self.can_execute:
                if TYPE_CHECKING:
                    assert isinstance(statement, Executable)
                self.execution_options = statement._execution_options
            self.string = self.process(self.statement, **compile_kwargs)
            if render_schema_translate:
                self.string = self.preparer._render_schema_translates(self.string, schema_translate_map)
            self.state = CompilerState.STRING_APPLIED
        else:
            self.state = CompilerState.NO_STATEMENT
        self._gen_time = perf_counter()

    def __init_subclass__(cls) -> None:
        cls._init_compiler_cls()
        return super().__init_subclass__()

    @classmethod
    def _init_compiler_cls(cls):
        pass

    def _execute_on_connection(self, connection, distilled_params, execution_options):
        if self.can_execute:
            return connection._execute_compiled(self, distilled_params, execution_options)
        else:
            raise exc.ObjectNotExecutableError(self.statement)

    def visit_unsupported_compilation(self, element, err, **kw):
        raise exc.UnsupportedCompilationError(self, type(element)) from err

    @property
    def sql_compiler(self):
        """Return a Compiled that is capable of processing SQL expressions.

        If this compiler is one, it would likely just return 'self'.

        """
        raise NotImplementedError()

    def process(self, obj: Visitable, **kwargs: Any) -> str:
        return obj._compiler_dispatch(self, **kwargs)

    def __str__(self) -> str:
        """Return the string text of the generated SQL or DDL."""
        if self.state is CompilerState.STRING_APPLIED:
            return self.string
        else:
            return ''

    def construct_params(self, params: Optional[_CoreSingleExecuteParams]=None, extracted_parameters: Optional[Sequence[BindParameter[Any]]]=None, escape_names: bool=True) -> Optional[_MutableCoreSingleExecuteParams]:
        """Return the bind params for this compiled object.

        :param params: a dict of string/object pairs whose values will
                       override bind values compiled in to the
                       statement.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Return the bind params for this compiled object."""
        return self.construct_params()