import argparse
import dataclasses
import fnmatch
import functools
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import AbstractSet
from typing import Callable
from typing import Dict
from typing import final
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from _pytest import nodes
import _pytest._code
from _pytest.config import Config
from _pytest.config import directory_arg
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.config.compat import PathAwareHookProxy
from _pytest.fixtures import FixtureManager
from _pytest.outcomes import exit
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import safe_exists
from _pytest.pathlib import scandir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.runner import collect_one_node
from _pytest.runner import SetupState
from _pytest.warning_types import PytestWarning
def _collect_path(self, path: Path, path_cache: Dict[Path, Sequence[nodes.Collector]]) -> Sequence[nodes.Collector]:
    """Create a Collector for the given path.

        `path_cache` makes it so the same Collectors are returned for the same
        path.
        """
    if path in path_cache:
        return path_cache[path]
    if path.is_dir():
        ihook = self.gethookproxy(path.parent)
        col: Optional[nodes.Collector] = ihook.pytest_collect_directory(path=path, parent=self)
        cols: Sequence[nodes.Collector] = (col,) if col is not None else ()
    elif path.is_file():
        ihook = self.gethookproxy(path)
        cols = ihook.pytest_collect_file(file_path=path, parent=self)
    else:
        cols = ()
    path_cache[path] = cols
    return cols