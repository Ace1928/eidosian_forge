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
@mock.patch('cinderclient.shell.CinderClientArgumentParser.exit')
def test_snapshot_create_pre_3_66_with_naked_force(self, mock_exit):
    mock_exit.side_effect = Exception('mock exit')
    try:
        self.run_command('--os-volume-api-version 3.65 snapshot-create --force 123456')
    except Exception as e:
        self.assertEqual('mock exit', str(e))
    exit_code = mock_exit.call_args.args[0]
    self.assertEqual(2, exit_code)