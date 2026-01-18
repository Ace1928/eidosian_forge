import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
def test_setup_sets_when_missing(self):
    fixture = EnvironmentVariable('FIXTURES_TEST_VAR', 'bar')
    os.environ.pop('FIXTURES_TEST_VAR', '')
    self.useFixture(fixture)
    self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))