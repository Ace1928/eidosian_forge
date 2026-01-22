import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
class ModuleNamespace(Namespace):
    """A :class:`.Namespace` specific to a Python module instance."""

    def __init__(self, name, context, module, callables=None, inherits=None, populate_self=True, calling_uri=None):
        self.name = name
        self.context = context
        self.inherits = inherits
        if callables is not None:
            self.callables = {c.__name__: c for c in callables}
        mod = __import__(module)
        for token in module.split('.')[1:]:
            mod = getattr(mod, token)
        self.module = mod

    @property
    def filename(self):
        """The path of the filesystem file used for this
        :class:`.Namespace`'s module or template.
        """
        return self.module.__file__

    def _get_star(self):
        if self.callables:
            for key in self.callables:
                yield (key, self.callables[key])
        for key in dir(self.module):
            if key[0] != '_':
                callable_ = getattr(self.module, key)
                if callable(callable_):
                    yield (key, functools.partial(callable_, self.context))

    def __getattr__(self, key):
        if key in self.callables:
            val = self.callables[key]
        elif hasattr(self.module, key):
            callable_ = getattr(self.module, key)
            val = functools.partial(callable_, self.context)
        elif self.inherits:
            val = getattr(self.inherits, key)
        else:
            raise AttributeError("Namespace '%s' has no member '%s'" % (self.name, key))
        setattr(self, key, val)
        return val