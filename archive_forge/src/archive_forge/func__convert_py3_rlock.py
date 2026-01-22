from __future__ import annotations
import sys
import eventlet
def _convert_py3_rlock(old, tid):
    """
    Convert a normal RLock to one implemented in Python.

    This is necessary to make RLocks work with eventlet, but also introduces
    bugs, e.g. https://bugs.python.org/issue13697.  So more of a downgrade,
    really.
    """
    import threading
    from eventlet.green.thread import allocate_lock
    new = threading._PyRLock()
    if not hasattr(new, '_block') or not hasattr(new, '_owner'):
        raise RuntimeError('INTERNAL BUG. Perhaps you are using a major version ' + 'of Python that is unsupported by eventlet? Please file a bug ' + 'at https://github.com/eventlet/eventlet/issues/new')
    new._block = allocate_lock()
    acquired = False
    while old._is_owned():
        old.release()
        new.acquire()
        acquired = True
    if old._is_owned():
        new.acquire()
        acquired = True
    if acquired:
        new._owner = tid
    return new