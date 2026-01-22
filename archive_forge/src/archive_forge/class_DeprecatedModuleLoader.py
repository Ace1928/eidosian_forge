import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
class DeprecatedModuleLoader(importlib.abc.Loader):
    """A Loader for deprecated modules.

    It wraps an existing Loader instance, to which it delegates the loading. On top of that
    it ensures that the sys.modules cache has both the deprecated module's name and the
    new module's name pointing to the same exact ModuleType instance.

    Args:
        loader: the loader to be wrapped
        old_module_name: the deprecated module's fully qualified name
        new_module_name: the new module's fully qualified name
    """

    def __init__(self, loader: Any, old_module_name: str, new_module_name: str):
        """A module loader that uses an existing module loader and intercepts
        the execution of a module.
        """
        self.loader = loader
        if hasattr(loader, 'exec_module'):
            self.exec_module = self._wrap_exec_module(loader.exec_module)
        if hasattr(loader, 'load_module'):
            self.load_module = self._wrap_load_module(loader.load_module)
        if hasattr(loader, 'create_module'):
            self.create_module = loader.create_module
        self.old_module_name = old_module_name
        self.new_module_name = new_module_name

    def module_repr(self, module: ModuleType) -> str:
        return self.loader.module_repr(module)

    def _wrap_load_module(self, method: Any) -> Any:

        def load_module(fullname: str) -> ModuleType:
            assert fullname == self.old_module_name, f'DeprecatedModuleLoader for {self.old_module_name} was asked to load {fullname}'
            if self.new_module_name in sys.modules:
                sys.modules[self.old_module_name] = sys.modules[self.new_module_name]
                return sys.modules[self.old_module_name]
            method(self.new_module_name)
            assert self.new_module_name in sys.modules, f'Wrapped loader {self.loader} was expected to insert {self.new_module_name} in sys.modules but it did not.'
            sys.modules[self.old_module_name] = sys.modules[self.new_module_name]
            return sys.modules[self.old_module_name]
        return load_module

    def _wrap_exec_module(self, method: Any) -> Any:

        def exec_module(module: ModuleType) -> None:
            assert module.__name__ == self.old_module_name, f'DeprecatedModuleLoader for {self.old_module_name} was asked to load {module.__name__}'
            if self.new_module_name in sys.modules:
                sys.modules[self.old_module_name] = sys.modules[self.new_module_name]
                return
            sys.modules[self.old_module_name] = module
            sys.modules[self.new_module_name] = module
            try:
                return method(module)
            except BaseException:
                del sys.modules[self.new_module_name]
                del sys.modules[self.old_module_name]
                raise
        return exec_module