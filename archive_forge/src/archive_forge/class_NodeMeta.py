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
class NodeMeta(abc.ABCMeta):
    """Metaclass used by :class:`Node` to enforce that direct construction raises
    :class:`Failed`.

    This behaviour supports the indirection introduced with :meth:`Node.from_parent`,
    the named constructor to be used instead of direct construction. The design
    decision to enforce indirection with :class:`NodeMeta` was made as a
    temporary aid for refactoring the collection tree, which was diagnosed to
    have :class:`Node` objects whose creational patterns were overly entangled.
    Once the refactoring is complete, this metaclass can be removed.

    See https://github.com/pytest-dev/pytest/projects/3 for an overview of the
    progress on detangling the :class:`Node` classes.
    """

    def __call__(cls, *k, **kw) -> NoReturn:
        msg = 'Direct construction of {name} has been deprecated, please use {name}.from_parent.\nSee https://docs.pytest.org/en/stable/deprecations.html#node-construction-changed-to-node-from-parent for more details.'.format(name=f'{cls.__module__}.{cls.__name__}')
        fail(msg, pytrace=False)

    def _create(cls: Type[_T], *k, **kw) -> _T:
        try:
            return super().__call__(*k, **kw)
        except TypeError:
            sig = signature(getattr(cls, '__init__'))
            known_kw = {k: v for k, v in kw.items() if k in sig.parameters}
            from .warning_types import PytestDeprecationWarning
            warnings.warn(PytestDeprecationWarning(f'{cls} is not using a cooperative constructor and only takes {set(known_kw)}.\nSee https://docs.pytest.org/en/stable/deprecations.html#constructors-of-custom-pytest-node-subclasses-should-take-kwargs for more details.'))
            return super().__call__(*k, **known_kw)