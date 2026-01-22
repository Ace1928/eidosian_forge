from pkgutil import extend_path
import sys
import os
import importlib
import types
from . import _gi
from ._gi import _API  # noqa: F401
from ._gi import Repository
from ._gi import PyGIDeprecationWarning  # noqa: F401
from ._gi import PyGIWarning  # noqa: F401
class _DummyStaticModule(types.ModuleType):
    __path__ = None

    def __getattr__(self, name):
        raise AttributeError(_static_binding_error)