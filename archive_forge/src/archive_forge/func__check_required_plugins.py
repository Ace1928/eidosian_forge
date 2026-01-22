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
def _check_required_plugins(plugins: list[Plugin], expected: frozenset[str]) -> None:
    plugin_names = {utils.normalize_pypi_name(plugin.package) for plugin in plugins}
    expected_names = {utils.normalize_pypi_name(name) for name in expected}
    missing_plugins = expected_names - plugin_names
    if missing_plugins:
        raise ExecutionError(f'required plugins were not installed!\n- installed: {', '.join(sorted(plugin_names))}\n- expected: {', '.join(sorted(expected_names))}\n- missing: {', '.join(sorted(missing_plugins))}')