import functools
import inspect
import wrapt
from debtcollector import _utils
def _wrap_it(old_init, out_message):

    @functools.wraps(old_init, assigned=_utils.get_assigned(old_init))
    def new_init(self, *args, **kwargs):
        _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
        return old_init(self, *args, **kwargs)
    return new_init