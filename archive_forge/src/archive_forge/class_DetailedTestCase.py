import unittest
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures import TestWithFixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
class DetailedTestCase(TestWithFixtures, testtools.TestCase):

    def setUp(self):
        super(DetailedTestCase, self).setUp()
        self.useFixture(broken_fixture)

    def test(self):
        pass