import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
def _undefer(self):
    """
        Return a non-deferred ObjectProxy referencing the same object
        """
    return self._parent.__getattr__(self._attributes[-1], _deferGetattr=False)