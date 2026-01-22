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
class PluginOptions(NamedTuple):
    """Options related to plugin loading."""
    local_plugin_paths: tuple[str, ...]
    enable_extensions: frozenset[str]
    require_plugins: frozenset[str]

    @classmethod
    def blank(cls) -> PluginOptions:
        """Make a blank PluginOptions, mostly used for tests."""
        return cls(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset())