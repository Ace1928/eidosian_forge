from __future__ import annotations
import ast
import sys
import types
from collections.abc import Callable, Iterable
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from os import PathLike
from types import CodeType, ModuleType, TracebackType
from typing import Sequence, TypeVar
from unittest.mock import patch
from ._config import global_config
from ._transformer import TypeguardTransformer
class ImportHookManager:
    """
    A handle that can be used to uninstall the Typeguard import hook.
    """

    def __init__(self, hook: MetaPathFinder):
        self.hook = hook

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        self.uninstall()

    def uninstall(self) -> None:
        """Uninstall the import hook."""
        try:
            sys.meta_path.remove(self.hook)
        except ValueError:
            pass