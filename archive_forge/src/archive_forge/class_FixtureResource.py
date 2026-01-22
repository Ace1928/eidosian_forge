import heapq
import inspect
import unittest
from pbr.version import VersionInfo
class FixtureResource(TestResourceManager):
    """A TestResourceManager that decorates a ``fixtures.Fixture``.

    The fixture has its setUp and cleanUp called as expected, and
    reset is called between uses.

    Due to the API of fixtures, dependency_resources are not
    accessible to the wrapped fixture. However, if you are using
    resource optimisation, you should wrap any dependencies in a
    FixtureResource and set the resources attribute appropriately.
    Note that when this is done, testresources will take care of
    calling setUp and cleanUp on the dependency fixtures and so
    the fixtures should not implicitly setUp or cleanUp their
    dependencies (that have been mapped).

    See the ``fixtures`` documentation for information on managing
    dependencies within the ``fixtures`` API.

    :ivar fixture: The wrapped fixture.
    """

    def __init__(self, fixture):
        """Create a FixtureResource

        :param fixture: The fixture to wrap.
        """
        super(FixtureResource, self).__init__()
        self.fixture = fixture

    def clean(self, resource):
        resource.cleanUp()

    def make(self, dependency_resources):
        self.fixture.setUp()
        return self.fixture

    def _reset(self, resource, dependency_resources):
        self.fixture.reset()
        return self.fixture

    def isDirty(self):
        return True
    _dirty = property(lambda _: True, lambda _, _1: None)