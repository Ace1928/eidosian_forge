import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_show_usage_with_help_flag(self):
    self.assertRaises(SystemExit, self.barbican.run, ['-h'])
    self.assertIn('usage', self.captured_stdout.getvalue())