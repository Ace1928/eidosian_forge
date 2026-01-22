import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle
class LogMixin(object):
    """Mixin class that adds logging functionality to another class.

    You can define a new class that subclasses from ``LogMixin`` as well as
    other base classes through multiple inheritance.
    All instances of that class will have a ``log`` property that returns
    a ``logging.Logger`` named after their respective ``<module>.<class>``.

    For example:

    >>> class BaseClass(object):
    ...     pass
    >>> class MyClass(LogMixin, BaseClass):
    ...     pass
    >>> a = MyClass()
    >>> isinstance(a.log, logging.Logger)
    True
    >>> print(a.log.name)
    fontTools.misc.loggingTools.MyClass
    >>> class AnotherClass(MyClass):
    ...     pass
    >>> b = AnotherClass()
    >>> isinstance(b.log, logging.Logger)
    True
    >>> print(b.log.name)
    fontTools.misc.loggingTools.AnotherClass
    """

    @property
    def log(self):
        if not hasattr(self, '_log'):
            name = '.'.join((self.__class__.__module__, self.__class__.__name__))
            self._log = logging.getLogger(name)
        return self._log