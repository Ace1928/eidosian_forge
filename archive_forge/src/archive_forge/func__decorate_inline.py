import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _decorate_inline(context, fn):

    def decorate_render(render_fn):
        dec = fn(render_fn)

        def go(*args, **kw):
            return dec(context, *args, **kw)
        return go
    return decorate_render