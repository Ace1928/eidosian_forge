import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _ctx_name_callback():
    """
    We don't use threading.current_thread() because it will deadlock if
    called when profiling threading._active_limbo_lock.acquire().
    See: #Issue48.
    """
    try:
        current_thread = threading._active[get_ident()]
        return current_thread.__class__.__name__
    except KeyError:
        return None