import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _kwargs_for_include(callable_, data, **kwargs):
    argspec = compat.inspect_getargspec(callable_)
    namedargs = argspec[0] + [v for v in argspec[1:3] if v is not None]
    for arg in namedargs:
        if arg != 'context' and arg in data and (arg not in kwargs):
            kwargs[arg] = data[arg]
    return kwargs