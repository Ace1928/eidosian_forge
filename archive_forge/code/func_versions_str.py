from __future__ import annotations
import configparser
import importlib.metadata
import inspect
import itertools
import logging
import sys
from typing import Any
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from flake8 import utils
from flake8.defaults import VALID_CODE_PREFIX
from flake8.exceptions import ExecutionError
from flake8.exceptions import FailedToLoadPlugin
def versions_str(self) -> str:
    """Return a user-displayed list of plugin versions."""
    return ', '.join(sorted({f'{loaded.plugin.package}: {loaded.plugin.version}' for loaded in self.all_plugins() if loaded.plugin.package not in {'flake8', 'local'}}))