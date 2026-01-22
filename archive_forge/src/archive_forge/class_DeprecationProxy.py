import collections
import functools
import importlib
import inspect
import sys
import types
import warnings
from humanfriendly.text import format
class DeprecationProxy(types.ModuleType):
    """Emit deprecation warnings for imports that should be updated."""

    def __init__(self, module, aliases):
        """
        Initialize an :class:`DeprecationProxy` object.

        :param module: The original module object.
        :param aliases: A dictionary of aliases.
        """
        super(DeprecationProxy, self).__init__(name=module.__name__)
        self.module = module
        self.aliases = aliases

    def __getattr__(self, name):
        """
        Override module attribute lookup.

        :param name: The name to look up (a string).
        :returns: The attribute value.
        """
        target = self.aliases.get(name)
        if target is not None:
            warnings.warn(format('%s.%s was moved to %s, please update your imports', self.module.__name__, name, target), category=DeprecationWarning, stacklevel=2)
            return self.resolve(target)
        value = getattr(self.module, name, None)
        if value is not None:
            return value
        raise AttributeError(format("module '%s' has no attribute '%s'", self.module.__name__, name))

    def resolve(self, target):
        """
        Look up the target of an alias.

        :param target: The fully qualified dotted path (a string).
        :returns: The value of the given target.
        """
        module_name, _, member = target.rpartition('.')
        module = importlib.import_module(module_name)
        return getattr(module, member)