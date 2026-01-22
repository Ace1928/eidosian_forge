from __future__ import absolute_import
import itertools
import sys
from weakref import ref
@classmethod
def _exitfunc(cls):
    reenable_gc = False
    try:
        if cls._registry:
            import gc
            if gc.isenabled():
                reenable_gc = True
                gc.disable()
            pending = None
            while True:
                if pending is None or backport_finalize._dirty:
                    pending = cls._select_for_exit()
                    backport_finalize._dirty = False
                if not pending:
                    break
                f = pending.pop()
                try:
                    f()
                except Exception:
                    sys.excepthook(*sys.exc_info())
                assert f not in cls._registry
    finally:
        backport_finalize._shutdown = True
        if reenable_gc:
            gc.enable()