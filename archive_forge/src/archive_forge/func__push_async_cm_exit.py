import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def _push_async_cm_exit(self, cm, cm_exit):
    """Helper to correctly register coroutine function to __aexit__
        method."""
    _exit_wrapper = self._create_async_exit_wrapper(cm, cm_exit)
    self._push_exit_callback(_exit_wrapper, False)