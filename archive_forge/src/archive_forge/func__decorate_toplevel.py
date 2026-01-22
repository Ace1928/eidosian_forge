import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _decorate_toplevel(fn):

    def decorate_render(render_fn):

        def go(context, *args, **kw):

            def y(*args, **kw):
                return render_fn(context, *args, **kw)
            try:
                y.__name__ = render_fn.__name__[7:]
            except TypeError:
                pass
            return fn(y)(context, *args, **kw)
        return go
    return decorate_render