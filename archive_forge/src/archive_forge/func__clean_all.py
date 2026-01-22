import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _clean_all(self, resource, result):
    """Clean the dependencies from resource, and then resource itself."""
    self._call_result_method_if_exists(result, 'startCleanResource', self)
    self.clean(resource)
    for name, manager in self.resources:
        manager.finishedWith(getattr(resource, name))
    self._call_result_method_if_exists(result, 'stopCleanResource', self)