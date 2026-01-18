import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
def test_setup_ignores_missing(self):
    fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
    os.environ.pop('FIXTURES_TEST_VAR', '')
    self.useFixture(fixture)
    self.assertEqual(None, os.environ.get('FIXTURES_TEST_VAR'))