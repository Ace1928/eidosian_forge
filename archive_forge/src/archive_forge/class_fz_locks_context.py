from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_locks_context(object):
    """
    Locking functions

    MuPDF is kept deliberately free of any knowledge of particular
    threading systems. As such, in order for safe multi-threaded
    operation, we rely on callbacks to client provided functions.

    A client is expected to provide FZ_LOCK_MAX number of mutexes,
    and a function to lock/unlock each of them. These may be
    recursive mutexes, but do not have to be.

    If a client does not intend to use multiple threads, then it
    may pass NULL instead of a lock structure.

    In order to avoid deadlocks, we have one simple rule
    internally as to how we use locks: We can never take lock n
    when we already hold any lock i, where 0 <= i <= n. In order
    to verify this, we have some debugging code, that can be
    enabled by defining FITZ_DEBUG_LOCKING.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    user = property(_mupdf.fz_locks_context_user_get, _mupdf.fz_locks_context_user_set)
    lock = property(_mupdf.fz_locks_context_lock_get, _mupdf.fz_locks_context_lock_set)
    unlock = property(_mupdf.fz_locks_context_unlock_get, _mupdf.fz_locks_context_unlock_set)

    def __init__(self):
        _mupdf.fz_locks_context_swiginit(self, _mupdf.new_fz_locks_context())
    __swig_destroy__ = _mupdf.delete_fz_locks_context