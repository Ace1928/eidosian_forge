from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import logging
import os
import sys
import threading
from googlecloudsdk.core.util import encoding
def _pairs(self, prefix):
    """Generates `(key, value)` pairs from the config module matching prefix.

    Args:
      prefix: A prefix string ending in `_`, for example: `mylib_`.

    Yields:
      `(key, value)` pairs, where `key` is the configuration name with the
      prefix removed, and `value` is the corresponding value.
    """
    self._lock.acquire()
    try:
        mapping = getattr(self._module, '__dict__', None)
        if not mapping:
            return
        items = list(mapping.items())
    finally:
        self._lock.release()
    nskip = len(prefix)
    for key, value in items:
        if key.startswith(prefix):
            yield (key[nskip:], value)