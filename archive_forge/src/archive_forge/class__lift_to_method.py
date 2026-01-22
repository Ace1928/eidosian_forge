import functools
import inspect
import types
from typing import Type, Callable
class _lift_to_method:
    """A decorator that ensures that an input callable object implements ``__get__``.  It is
    returned unchanged if so, otherwise it is turned into the default implementation for functions,
    which makes them bindable to instances.

    Python-space functions and lambdas already have this behaviour, but builtins like ``print``
    don't; using this class allows us to do::

        wrap_method(MyClass, "maybe_mutates_arguments", before=print, after=print)

    to simply print all the arguments on entry and exit of the function, which otherwise wouldn't be
    valid, since ``print`` isn't a descriptor.
    """
    __slots__ = ('_method',)

    def __new__(cls, method):
        if hasattr(method, '__get__'):
            return method
        return super().__new__(cls)

    def __init__(self, method):
        if method is self:
            return
        self._method = method

    def __get__(self, obj, objtype):
        if obj is None:
            return self._method
        return types.MethodType(self._method, obj)