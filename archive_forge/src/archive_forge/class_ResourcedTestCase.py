import heapq
import inspect
import unittest
from pbr.version import VersionInfo
class ResourcedTestCase(unittest.TestCase):
    """A TestCase parent or utility that enables cross-test resource usage.

    ResourcedTestCase is a thin wrapper around the
    testresources.setUpResources and testresources.tearDownResources helper
    functions. It should be trivially reimplemented where a different base
    class is neded, or you can use multiple inheritance and call into
    ResourcedTestCase.setUpResources and ResourcedTestCase.tearDownResources
    from your setUp and tearDown (or whatever cleanup idiom is used).

    :ivar resources: A list of (name, resource) pairs, where 'resource' is a
        subclass of `TestResourceManager` and 'name' is the name of the
        attribute that the resource should be stored on.
    """
    resources = []

    def setUp(self):
        super(ResourcedTestCase, self).setUp()
        self.setUpResources()

    def setUpResources(self):
        setUpResources(self, self.resources, _get_result())

    def tearDown(self):
        self.tearDownResources()
        super(ResourcedTestCase, self).tearDown()

    def tearDownResources(self):
        tearDownResources(self, self.resources, _get_result())