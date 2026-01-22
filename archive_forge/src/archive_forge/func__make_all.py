import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _make_all(self, result):
    """Make the dependencies of this resource and this resource."""
    self._call_result_method_if_exists(result, 'startMakeResource', self)
    dependency_resources = {}
    for name, resource in self.resources:
        dependency_resources[name] = resource.getResource()
    resource = self.make(dependency_resources)
    for name, value in dependency_resources.items():
        setattr(resource, name, value)
    self._call_result_method_if_exists(result, 'stopMakeResource', self)
    return resource