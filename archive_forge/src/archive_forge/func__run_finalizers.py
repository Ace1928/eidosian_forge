import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def _run_finalizers(minpriority=None):
    """
    Run all finalizers whose exit priority is not None and at least minpriority

    Finalizers with highest priority are called first; finalizers with
    the same priority will be called in reverse order of creation.
    """
    if _finalizer_registry is None:
        return
    if minpriority is None:
        f = lambda p: p[0] is not None
    else:
        f = lambda p: p[0] is not None and p[0] >= minpriority
    keys = [key for key in list(_finalizer_registry) if f(key)]
    keys.sort(reverse=True)
    for key in keys:
        finalizer = _finalizer_registry.get(key)
        if finalizer is not None:
            sub_debug('calling %s', finalizer)
            try:
                finalizer()
            except Exception:
                import traceback
                traceback.print_exc()
    if minpriority is None:
        _finalizer_registry.clear()