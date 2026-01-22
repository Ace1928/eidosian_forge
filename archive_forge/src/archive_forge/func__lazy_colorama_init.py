import os
import time
from contextlib import contextmanager
from typing import Callable, Optional
def _lazy_colorama_init():
    """
            Lazily init colorama if necessary, not to screw up stdout is
            debug not enabled.

            This version of the function does init colorama.
            """
    global _inited
    if not _inited:
        initialise.atexit_done = True
        try:
            init(strip=False)
        except Exception:
            pass
    _inited = True