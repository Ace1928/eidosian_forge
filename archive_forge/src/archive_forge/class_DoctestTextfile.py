import bdb
from contextlib import contextmanager
import functools
import inspect
import os
from pathlib import Path
import platform
import sys
import traceback
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import fixture
from _pytest.fixtures import TopRequest
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import OutcomeException
from _pytest.outcomes import skip
from _pytest.pathlib import fnmatch_ex
from _pytest.python import Module
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
class DoctestTextfile(Module):
    obj = None

    def collect(self) -> Iterable[DoctestItem]:
        import doctest
        encoding = self.config.getini('doctest_encoding')
        text = self.path.read_text(encoding)
        filename = str(self.path)
        name = self.path.name
        globs = {'__name__': '__main__'}
        optionflags = get_optionflags(self.config)
        runner = _get_runner(verbose=False, optionflags=optionflags, checker=_get_checker(), continue_on_failure=_get_continue_on_failure(self.config))
        parser = doctest.DocTestParser()
        test = parser.get_doctest(text, globs, name, filename, 0)
        if test.examples:
            yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)