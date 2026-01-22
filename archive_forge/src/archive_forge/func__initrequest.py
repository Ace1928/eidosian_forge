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
def _initrequest(self) -> None:
    self.funcargs: Dict[str, object] = {}
    self._request = TopRequest(self, _ispytest=True)