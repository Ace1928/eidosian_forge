import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def _push_cm_exit(self, cm, cm_exit):
    """Helper to correctly register callbacks to __exit__ methods."""
    _exit_wrapper = self._create_exit_wrapper(cm, cm_exit)
    self._push_exit_callback(_exit_wrapper, True)