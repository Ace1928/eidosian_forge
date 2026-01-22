from keystoneauth1 import fixture as ksa_fixture
from requests_mock.contrib import fixture
from openstackclient.tests.unit import test_shell
from openstackclient.tests.unit import utils
class TestInteg(utils.TestCase):

    def setUp(self):
        super(TestInteg, self).setUp()
        self.requests_mock = self.useFixture(fixture.Fixture())