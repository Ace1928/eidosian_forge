import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
class TestEnvironmentVariable(testtools.TestCase, TestWithFixtures):

    def test_setup_ignores_missing(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
        os.environ.pop('FIXTURES_TEST_VAR', '')
        self.useFixture(fixture)
        self.assertEqual(None, os.environ.get('FIXTURES_TEST_VAR'))

    def test_setup_sets_when_missing(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR', 'bar')
        os.environ.pop('FIXTURES_TEST_VAR', '')
        self.useFixture(fixture)
        self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))

    def test_setup_deletes(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
        os.environ['FIXTURES_TEST_VAR'] = 'foo'
        self.useFixture(fixture)
        self.assertEqual(None, os.environ.get('FIXTURES_TEST_VAR'))

    def test_setup_overrides(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR', 'bar')
        os.environ['FIXTURES_TEST_VAR'] = 'foo'
        self.useFixture(fixture)
        self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))

    def test_cleanup_deletes_when_missing(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
        os.environ.pop('FIXTURES_TEST_VAR', '')
        with fixture:
            os.environ['FIXTURES_TEST_VAR'] = 'foo'
        self.assertEqual(None, os.environ.get('FIXTURES_TEST_VAR'))

    def test_cleanup_deletes_when_set(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR', 'bar')
        os.environ.pop('FIXTURES_TEST_VAR', '')
        with fixture:
            os.environ['FIXTURES_TEST_VAR'] = 'foo'
        self.assertEqual(None, os.environ.get('FIXTURES_TEST_VAR'))

    def test_cleanup_restores_when_missing(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
        os.environ['FIXTURES_TEST_VAR'] = 'bar'
        with fixture:
            os.environ.pop('FIXTURES_TEST_VAR', '')
        self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))

    def test_cleanup_restores_when_set(self):
        fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
        os.environ['FIXTURES_TEST_VAR'] = 'bar'
        with fixture:
            os.environ['FIXTURES_TEST_VAR'] = 'quux'
        self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))