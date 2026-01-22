import time
import types
from ..trace import mutter
from ..transport import decorator
def _make_hook(hookname):

    def _hook(relpath, *args, **kw):
        return self._log_and_call(hookname, relpath, *args, **kw)
    return _hook