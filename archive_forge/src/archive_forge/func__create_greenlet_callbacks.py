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
def _create_greenlet_callbacks():
    """
    Returns two functions:
    - one that can identify unique greenlets. Identity of a greenlet
      cannot be reused once a greenlet dies. 'id(greenlet)' cannot be used because
      'id' returns an identifier that can be reused once a greenlet object is garbage
      collected.
    - one that can return the name of the greenlet class used to spawn the greenlet
    """
    try:
        from greenlet import getcurrent
    except ImportError as exc:
        raise YappiError(f"'greenlet' import failed with: {repr(exc)}")

    def _get_greenlet_id():
        curr_greenlet = getcurrent()
        id_ = getattr(curr_greenlet, '_yappi_tid', None)
        if id_ is None:
            id_ = GREENLET_COUNTER()
            curr_greenlet._yappi_tid = id_
        return id_

    def _get_greenlet_name():
        return getcurrent().__class__.__name__
    return (_get_greenlet_id, _get_greenlet_name)