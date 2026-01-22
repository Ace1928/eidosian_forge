import time
import types
from ..trace import mutter
from ..transport import decorator
def _log_and_call(self, methodname, relpath, *args, **kwargs):
    if kwargs:
        kwargs_str = dict(kwargs)
    else:
        kwargs_str = ''
    mutter('%s %s %s %s' % (methodname, relpath, self._shorten(self._strip_tuple_parens(args)), kwargs_str))
    return self._call_and_log_result(methodname, (relpath,) + args, kwargs)