import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _ensure_safe(self):
    """Makes sure no numpy arrays pointing to internal buffers are active.

    This should be called from any function that will call a function on
    _interpreter that may reallocate memory e.g. invoke(), ...

    Raises:
      RuntimeError: If there exist numpy objects pointing to internal memory
        then we throw.
    """
    if not self._safe_to_run():
        raise RuntimeError('There is at least 1 reference to internal data\n      in the interpreter in the form of a numpy array or slice. Be sure to\n      only hold the function returned from tensor() if you are using raw\n      data access.')