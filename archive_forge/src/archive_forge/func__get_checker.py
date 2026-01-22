import bdb
import builtins
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import (
import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo, ReprFileLocation, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex, import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
def _get_checker() -> 'IPDoctestOutputChecker':
    """Return a IPDoctestOutputChecker subclass that supports some
    additional options:

    * ALLOW_UNICODE and ALLOW_BYTES options to ignore u'' and b''
      prefixes (respectively) in string literals. Useful when the same
      ipdoctest should run in Python 2 and Python 3.

    * NUMBER to ignore floating-point differences smaller than the
      precision of the literal number in the ipdoctest.

    An inner class is used to avoid importing "ipdoctest" at the module
    level.
    """
    global CHECKER_CLASS
    if CHECKER_CLASS is None:
        CHECKER_CLASS = _init_checker_class()
    return CHECKER_CLASS()