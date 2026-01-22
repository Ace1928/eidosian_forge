import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
class _ModuleContext:
    """Record of the location within a source tree"""

    def __init__(self, path, lineno=1, _source_info=None):
        self.path = path
        self.lineno = lineno
        if _source_info is not None:
            self._cls_to_lineno, self._str_to_lineno = _source_info

    @classmethod
    def from_module(cls, module):
        """Get new context from module object and parse source for linenos"""
        sourcepath = inspect.getsourcefile(module)
        relpath = os.path.relpath(sourcepath)
        return cls(relpath, _source_info=_parse_source(''.join(inspect.findsource(module)[0]), module.__file__))

    def from_class(self, cls):
        """Get new context with same details but lineno of class in source"""
        try:
            lineno = self._cls_to_lineno[cls.__name__]
        except (AttributeError, KeyError):
            mutter('Definition of %r not found in %r', cls, self.path)
            return self
        return self.__class__(self.path, lineno, (self._cls_to_lineno, self._str_to_lineno))

    def from_string(self, string):
        """Get new context with same details but lineno of string in source"""
        try:
            lineno = self._str_to_lineno[string]
        except (AttributeError, KeyError):
            mutter('String %r not found in %r', string[:20], self.path)
            return self
        return self.__class__(self.path, lineno, (self._cls_to_lineno, self._str_to_lineno))