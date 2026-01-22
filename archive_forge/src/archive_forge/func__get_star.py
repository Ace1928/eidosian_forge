import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _get_star(self):
    if self.callables:
        for key in self.callables:
            yield (key, self.callables[key])
    for key in dir(self.module):
        if key[0] != '_':
            callable_ = getattr(self.module, key)
            if callable(callable_):
                yield (key, functools.partial(callable_, self.context))