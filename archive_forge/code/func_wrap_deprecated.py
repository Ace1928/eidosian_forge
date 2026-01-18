import inspect
import functools
import sys
import warnings
from eventlet.support import greenlets
def wrap_deprecated(old, new):

    def _resolve(s):
        return 'eventlet.' + s if '.' not in s else s
    msg = "{old} is deprecated and will be removed in next version. Use {new} instead.\nAutoupgrade: fgrep -rl '{old}' . |xargs -t sed --in-place='' -e 's/{old}/{new}/'\n".format(old=_resolve(old), new=_resolve(new))

    def wrapper(base):
        klass = None
        if inspect.isclass(base):

            class klass(base):
                pass
            klass.__name__ = base.__name__
            klass.__module__ = base.__module__

        @functools.wraps(base)
        def wrapped(*a, **kw):
            warnings.warn(msg, DeprecationWarning, stacklevel=5)
            return base(*a, **kw)
        if klass is not None:
            klass.__init__ = wrapped
            return klass
        return wrapped
    return wrapper