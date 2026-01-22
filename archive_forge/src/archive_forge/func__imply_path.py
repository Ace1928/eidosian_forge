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
def _imply_path(node_type: Type['Node'], path: Optional[Path], fspath: Optional[LEGACY_PATH]) -> Path:
    if fspath is not None:
        warnings.warn(NODE_CTOR_FSPATH_ARG.format(node_type_name=node_type.__name__), stacklevel=6)
    if path is not None:
        if fspath is not None:
            _check_path(path, fspath)
        return path
    else:
        assert fspath is not None
        return Path(fspath)