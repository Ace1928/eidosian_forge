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
def _compute_fixture_value(self, fixturedef: 'FixtureDef[object]') -> None:
    """Create a SubRequest based on "self" and call the execute method
        of the given FixtureDef object.

        This will force the FixtureDef object to throw away any previous
        results and compute a new fixture value, which will be stored into
        the FixtureDef object itself.
        """
    argname = fixturedef.argname
    funcitem = self._pyfuncitem
    try:
        callspec = funcitem.callspec
    except AttributeError:
        callspec = None
    if callspec is not None and argname in callspec.params:
        param = callspec.params[argname]
        param_index = callspec.indices[argname]
        scope = callspec._arg2scope[argname]
    else:
        param = NOTSET
        param_index = 0
        scope = fixturedef._scope
        has_params = fixturedef.params is not None
        fixtures_not_supported = getattr(funcitem, 'nofuncargs', False)
        if has_params and fixtures_not_supported:
            msg = f'{funcitem.name} does not support fixtures, maybe unittest.TestCase subclass?\nNode id: {funcitem.nodeid}\nFunction type: {type(funcitem).__name__}'
            fail(msg, pytrace=False)
        if has_params:
            frame = inspect.stack()[3]
            frameinfo = inspect.getframeinfo(frame[0])
            source_path = absolutepath(frameinfo.filename)
            source_lineno = frameinfo.lineno
            try:
                source_path_str = str(source_path.relative_to(funcitem.config.rootpath))
            except ValueError:
                source_path_str = str(source_path)
            msg = "The requested fixture has no parameter defined for test:\n    {}\n\nRequested fixture '{}' defined in:\n{}\n\nRequested here:\n{}:{}".format(funcitem.nodeid, fixturedef.argname, getlocation(fixturedef.func, funcitem.config.rootpath), source_path_str, source_lineno)
            fail(msg, pytrace=False)
    subrequest = SubRequest(self, scope, param, param_index, fixturedef, _ispytest=True)
    subrequest._check_scope(argname, self._scope, scope)
    try:
        fixturedef.execute(request=subrequest)
    finally:
        self._schedule_finalizers(fixturedef, subrequest)