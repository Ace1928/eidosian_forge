import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _set_with_template(self, t):
    self._with_template = t
    illegal_names = t.reserved_names.intersection(self._data)
    if illegal_names:
        raise exceptions.NameConflictError('Reserved words passed to render(): %s' % ', '.join(illegal_names))