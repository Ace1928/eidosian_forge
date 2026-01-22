from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
class _DeferredOr(_DeferredImportIndicatorBase):

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __bool__(self):
        return bool(self._a) or bool(self._b)