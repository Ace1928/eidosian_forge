import contextlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import MethodType, ModuleType
from typing import Any, Generator, Iterator, List, Optional, Sequence, Tuple, Union
from sphinx.util import logging
from sphinx.util.inspect import isboundmethod, safe_getattr
class MockLoader(Loader):
    """A loader for mocking."""

    def __init__(self, finder: 'MockFinder') -> None:
        super().__init__()
        self.finder = finder

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        logger.debug('[autodoc] adding a mock module as %s!', spec.name)
        self.finder.mocked_modules.append(spec.name)
        return _MockModule(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        pass