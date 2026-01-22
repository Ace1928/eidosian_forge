from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import logging
import os
import sys
import threading
from googlecloudsdk.core.util import encoding
def _update_configs(self):
    """Updates the configuration values.

    This clears the cached values, initializes the registry, and loads
    the configuration values from the config module.
    """
    self._lock.acquire()
    try:
        if self._initialized:
            self._clear_cache()
        self._registry.initialize()
        for key, value in self._registry._pairs(self._prefix):
            if key not in self._defaults:
                logging.warn('Configuration "%s" not recognized', self._prefix + key)
            else:
                self._overrides[key] = value
        self._initialized = True
    finally:
        self._lock.release()