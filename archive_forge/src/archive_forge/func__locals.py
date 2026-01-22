import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _locals(self, d):
    """Create a new :class:`.Context` with a copy of this
        :class:`.Context`'s current state,
        updated with the given dictionary.

        The :attr:`.Context.kwargs` collection remains
        unaffected.


        """
    if not d:
        return self
    c = self._copy()
    c._data.update(d)
    return c