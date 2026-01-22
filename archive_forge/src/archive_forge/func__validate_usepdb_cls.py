import argparse
import functools
import sys
import types
from typing import Any
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import unittest
from _pytest import outcomes
from _pytest._code import ExceptionInfo
from _pytest.config import Config
from _pytest.config import ConftestImportFailure
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.config.exceptions import UsageError
from _pytest.nodes import Node
from _pytest.reports import BaseReport
def _validate_usepdb_cls(value: str) -> Tuple[str, str]:
    """Validate syntax of --pdbcls option."""
    try:
        modname, classname = value.split(':')
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{value!r} is not in the format 'modname:classname'") from e
    return (modname, classname)