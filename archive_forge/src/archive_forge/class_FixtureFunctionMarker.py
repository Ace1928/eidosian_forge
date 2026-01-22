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
@dataclasses.dataclass(frozen=True)
class FixtureFunctionMarker:
    scope: 'Union[_ScopeName, Callable[[str, Config], _ScopeName]]'
    params: Optional[Tuple[object, ...]]
    autouse: bool = False
    ids: Optional[Union[Tuple[Optional[object], ...], Callable[[Any], Optional[object]]]] = None
    name: Optional[str] = None
    _ispytest: dataclasses.InitVar[bool] = False

    def __post_init__(self, _ispytest: bool) -> None:
        check_ispytest(_ispytest)

    def __call__(self, function: FixtureFunction) -> FixtureFunction:
        if inspect.isclass(function):
            raise ValueError('class fixtures not supported (maybe in the future)')
        if getattr(function, '_pytestfixturefunction', False):
            raise ValueError(f'@pytest.fixture is being applied more than once to the same function {function.__name__!r}')
        if hasattr(function, 'pytestmark'):
            warnings.warn(MARKED_FIXTURE, stacklevel=2)
        function = wrap_function_to_error_out_if_called_directly(function, self)
        name = self.name or function.__name__
        if name == 'request':
            location = getlocation(function)
            fail(f"'request' is a reserved word for fixtures, use another name:\n  {location}", pytrace=False)
        function._pytestfixturefunction = self
        return function