import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
def test_setup_overrides(self):
    fixture = EnvironmentVariable('FIXTURES_TEST_VAR', 'bar')
    os.environ['FIXTURES_TEST_VAR'] = 'foo'
    self.useFixture(fixture)
    self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))