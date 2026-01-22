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
class FixtureDef(Generic[FixtureValue]):
    """A container for a fixture definition.

    Note: At this time, only explicitly documented fields and methods are
    considered public stable API.
    """

    def __init__(self, config: Config, baseid: Optional[str], argname: str, func: '_FixtureFunc[FixtureValue]', scope: Union[Scope, _ScopeName, Callable[[str, Config], _ScopeName], None], params: Optional[Sequence[object]], unittest: bool=False, ids: Optional[Union[Tuple[Optional[object], ...], Callable[[Any], Optional[object]]]]=None, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self.baseid: Final = baseid or ''
        self.has_location: Final = baseid is not None
        self.func: Final = func
        self.argname: Final = argname
        if scope is None:
            scope = Scope.Function
        elif callable(scope):
            scope = _eval_scope_callable(scope, argname, config)
        if isinstance(scope, str):
            scope = Scope.from_user(scope, descr=f"Fixture '{func.__name__}'", where=baseid)
        self._scope: Final = scope
        self.params: Final = params
        self.ids: Final = ids
        self.argnames: Final = getfuncargnames(func, name=argname, is_method=unittest)
        self.unittest: Final = unittest
        self.cached_result: Optional[_FixtureCachedResult[FixtureValue]] = None
        self._finalizers: Final[List[Callable[[], object]]] = []

    @property
    def scope(self) -> _ScopeName:
        """Scope string, one of "function", "class", "module", "package", "session"."""
        return self._scope.value

    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        self._finalizers.append(finalizer)

    def finish(self, request: SubRequest) -> None:
        exceptions: List[BaseException] = []
        while self._finalizers:
            fin = self._finalizers.pop()
            try:
                fin()
            except BaseException as e:
                exceptions.append(e)
        node = request.node
        node.ihook.pytest_fixture_post_finalizer(fixturedef=self, request=request)
        self.cached_result = None
        self._finalizers.clear()
        if len(exceptions) == 1:
            raise exceptions[0]
        elif len(exceptions) > 1:
            msg = f'errors while tearing down fixture "{self.argname}" of {node}'
            raise BaseExceptionGroup(msg, exceptions[::-1])

    def execute(self, request: SubRequest) -> FixtureValue:
        for argname in self.argnames:
            fixturedef = request._get_active_fixturedef(argname)
            if argname != 'request':
                assert isinstance(fixturedef, FixtureDef)
                fixturedef.addfinalizer(functools.partial(self.finish, request=request))
        my_cache_key = self.cache_key(request)
        if self.cached_result is not None:
            cache_key = self.cached_result[1]
            if my_cache_key is cache_key:
                if self.cached_result[2] is not None:
                    exc = self.cached_result[2]
                    raise exc
                else:
                    result = self.cached_result[0]
                    return result
            self.finish(request)
            assert self.cached_result is None
        ihook = request.node.ihook
        result = ihook.pytest_fixture_setup(fixturedef=self, request=request)
        return result

    def cache_key(self, request: SubRequest) -> object:
        return request.param_index if not hasattr(request, 'param') else request.param

    def __repr__(self) -> str:
        return f'<FixtureDef argname={self.argname!r} scope={self.scope!r} baseid={self.baseid!r}>'