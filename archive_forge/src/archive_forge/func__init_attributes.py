import logging
import operator
from . import _cache
from .exception import NoMatches
def _init_attributes(self, namespace, propagate_map_exceptions=False, on_load_failure_callback=None):
    self.namespace = namespace
    self.propagate_map_exceptions = propagate_map_exceptions
    self._on_load_failure_callback = on_load_failure_callback