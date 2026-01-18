from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@ddt.data({'command': '--attach-status detached', 'expected': {'attach_status': 'detached'}}, {'command': '--state in-use --attach-status attached', 'expected': {'status': 'in-use', 'attach_status': 'attached'}}, {'command': '--reset-migration-status', 'expected': {'migration_status': 'none'}})
@ddt.unpack
def test_reset_state_volume_additional_status(self, command, expected):
    self.run_command('reset-state %s 1234' % command)
    expected = {'os-reset_status': expected}
    self.assert_called('POST', '/volumes/1234/action', body=expected)