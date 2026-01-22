import abc
from collections import defaultdict
from collections import deque
import dataclasses
import functools
import inspect
import os
from pathlib import Path
import sys
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
import _pytest
from _pytest import nodes
from _pytest._code import getfslineno
from _pytest._code.code import FormattedExcinfo
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import _PytestWrapper
from _pytest.compat import assert_never
from _pytest.compat import get_real_func
from _pytest.compat import get_real_method
from _pytest.compat import getfuncargnames
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.compat import safe_getattr
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.deprecated import YIELD_FIXTURE
from _pytest.mark import Mark
from _pytest.mark import ParameterSet
from _pytest.mark.structures import MarkDecorator
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import TEST_OUTCOME
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.scope import _ScopeName
from _pytest.scope import HIGH_SCOPES
from _pytest.scope import Scope
@final
class SubRequest(FixtureRequest):
    """The type of the ``request`` fixture in a fixture function requested
    (transitively) by a test function."""

    def __init__(self, request: FixtureRequest, scope: Scope, param: Any, param_index: int, fixturedef: 'FixtureDef[object]', *, _ispytest: bool=False) -> None:
        super().__init__(pyfuncitem=request._pyfuncitem, fixturename=fixturedef.argname, fixture_defs=request._fixture_defs, arg2fixturedefs=request._arg2fixturedefs, _ispytest=_ispytest)
        self._parent_request: Final[FixtureRequest] = request
        self._scope_field: Final = scope
        self._fixturedef: Final[FixtureDef[object]] = fixturedef
        if param is not NOTSET:
            self.param = param
        self.param_index: Final = param_index

    def __repr__(self) -> str:
        return f'<SubRequest {self.fixturename!r} for {self._pyfuncitem!r}>'

    @property
    def _scope(self) -> Scope:
        return self._scope_field

    @property
    def node(self):
        scope = self._scope
        if scope is Scope.Function:
            node: Optional[nodes.Node] = self._pyfuncitem
        elif scope is Scope.Package:
            node = get_scope_package(self._pyfuncitem, self._fixturedef)
        else:
            node = get_scope_node(self._pyfuncitem, scope)
        if node is None and scope is Scope.Class:
            node = self._pyfuncitem
        assert node, f'Could not obtain a node for scope "{scope}" for function {self._pyfuncitem!r}'
        return node

    def _check_scope(self, argname: str, invoking_scope: Scope, requested_scope: Scope) -> None:
        if argname == 'request':
            return
        if invoking_scope > requested_scope:
            text = '\n'.join(self._factorytraceback())
            fail(f'ScopeMismatch: You tried to access the {requested_scope.value} scoped fixture {argname} with a {invoking_scope.value} scoped request object, involved factories:\n{text}', pytrace=False)

    def _factorytraceback(self) -> List[str]:
        lines = []
        for fixturedef in self._get_fixturestack():
            factory = fixturedef.func
            fs, lineno = getfslineno(factory)
            if isinstance(fs, Path):
                session: Session = self._pyfuncitem.session
                p = bestrelpath(session.path, fs)
            else:
                p = fs
            lines.append('%s:%d:  def %s%s' % (p, lineno + 1, factory.__name__, inspect.signature(factory)))
        return lines

    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        self._fixturedef.addfinalizer(finalizer)

    def _schedule_finalizers(self, fixturedef: 'FixtureDef[object]', subrequest: 'SubRequest') -> None:
        if fixturedef.argname not in self._fixture_defs and fixturedef.argname not in self._pyfuncitem.fixturenames:
            fixturedef.addfinalizer(functools.partial(self._fixturedef.finish, request=self))
        super()._schedule_finalizers(fixturedef, subrequest)