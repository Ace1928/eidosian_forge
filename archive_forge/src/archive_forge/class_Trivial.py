from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
class Trivial(testresources.TestResource):

    def __init__(self, thing):
        testresources.TestResource.__init__(self)
        self.thing = thing

    def make(self, dependency_resources):
        return self.thing

    def clean(self, resource):
        pass