import stevedore
from testtools import matchers
from keystone.tests.unit import core as test
class TestEntryPoints(test.TestCase):

    def test_entry_point_middleware(self):
        """Assert that our list of expected middleware is present."""
        expected_names = ['cors', 'debug', 'request_id', 'sizelimit']
        em = stevedore.ExtensionManager('keystone.server_middleware')
        actual_names = [extension.name for extension in em]
        self.assertThat(actual_names, matchers.ContainsAll(expected_names))