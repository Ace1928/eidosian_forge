import functools
import types
import warnings
import importlib
import sys
from gi import PyGIDeprecationWarning
from gi._gi import CallableInfo, pygobject_new_full
from gi._constants import \
from pkgutil import extend_path
class OverridesProxyModule(types.ModuleType):
    """Wraps a introspection module and contains all overrides"""

    def __init__(self, introspection_module):
        super(OverridesProxyModule, self).__init__(introspection_module.__name__)
        self._introspection_module = introspection_module

    def __getattr__(self, name):
        return getattr(self._introspection_module, name)

    def __dir__(self):
        result = set(dir(self.__class__))
        result.update(self.__dict__.keys())
        result.update(dir(self._introspection_module))
        return sorted(result)

    def __repr__(self):
        return '<%s %r>' % (type(self).__name__, self._introspection_module)