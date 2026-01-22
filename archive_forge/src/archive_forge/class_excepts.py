from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
class excepts(object):
    """A wrapper around a function to catch exceptions and
    dispatch to a handler.

    This is like a functional try/except block, in the same way that
    ifexprs are functional if/else blocks.

    Examples
    --------
    >>> excepting = excepts(
    ...     ValueError,
    ...     lambda a: [1, 2].index(a),
    ...     lambda _: -1,
    ... )
    >>> excepting(1)
    0
    >>> excepting(3)
    -1

    Multiple exceptions and default except clause.

    >>> excepting = excepts((IndexError, KeyError), lambda a: a[0])
    >>> excepting([])
    >>> excepting([1])
    1
    >>> excepting({})
    >>> excepting({0: 1})
    1
    """

    def __init__(self, exc, func, handler=return_none):
        self.exc = exc
        self.func = func
        self.handler = handler

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except self.exc as e:
            return self.handler(e)

    @instanceproperty(classval=__doc__)
    def __doc__(self):
        from textwrap import dedent
        exc = self.exc
        try:
            if isinstance(exc, tuple):
                exc_name = '(%s)' % ', '.join(map(attrgetter('__name__'), exc))
            else:
                exc_name = exc.__name__
            return dedent('                A wrapper around {inst.func.__name__!r} that will except:\n                {exc}\n                and handle any exceptions with {inst.handler.__name__!r}.\n\n                Docs for {inst.func.__name__!r}:\n                {inst.func.__doc__}\n\n                Docs for {inst.handler.__name__!r}:\n                {inst.handler.__doc__}\n                ').format(inst=self, exc=exc_name)
        except AttributeError:
            return type(self).__doc__

    @property
    def __name__(self):
        exc = self.exc
        try:
            if isinstance(exc, tuple):
                exc_name = '_or_'.join(map(attrgetter('__name__'), exc))
            else:
                exc_name = exc.__name__
            return '%s_excepting_%s' % (self.func.__name__, exc_name)
        except AttributeError:
            return 'excepting'