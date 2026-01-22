import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def _push_exit_callback(self, callback, is_sync=True):
    self._exit_callbacks.append((is_sync, callback))