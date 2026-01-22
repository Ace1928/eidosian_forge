import abc
from functools import cached_property
from inspect import signature
import os
import pathlib
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
import pluggy
import _pytest._code
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest._code.code import Traceback
from _pytest.compat import LEGACY_PATH
from _pytest.config import Config
from _pytest.config import ConftestImportFailure
from _pytest.config.compat import _check_path
from _pytest.deprecated import NODE_CTOR_FSPATH_ARG
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import NodeKeywords
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.stash import Stash
from _pytest.warning_types import PytestWarning
def _check_item_and_collector_diamond_inheritance(self) -> None:
    """
        Check if the current type inherits from both File and Collector
        at the same time, emitting a warning accordingly (#8447).
        """
    cls = type(self)
    attr_name = '_pytest_diamond_inheritance_warning_shown'
    if getattr(cls, attr_name, False):
        return
    setattr(cls, attr_name, True)
    problems = ', '.join((base.__name__ for base in cls.__bases__ if issubclass(base, Collector)))
    if problems:
        warnings.warn(f'{cls.__name__} is an Item subclass and should not be a collector, however its bases {problems} are collectors.\nPlease split the Collectors and the Item into separate node types.\nPytest Doc example: https://docs.pytest.org/en/latest/example/nonpython.html\nexample pull request on a plugin: https://github.com/asmeurer/pytest-flakes/pull/40/', PytestWarning)