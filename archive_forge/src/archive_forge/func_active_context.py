import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
@classmethod
def active_context(cls) -> Optional['AnnotatedQueue']:
    """Returns the currently active queuing context."""
    return cls._active_contexts[-1] if cls.recording() else None