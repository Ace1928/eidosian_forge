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
@mock.patch('cinderclient.utils.find_resource')
def test_snapshot_create_3_66_not_supported(self, mock_find_vol):
    mock_find_vol.return_value = volumes.Volume(self, {'id': '123456'}, loaded=True)
    self.run_command_with_server_api_max('3.64', '--os-volume-api-version 3.66 snapshot-create 123456')
    pre_3_66_request_body = {'snapshot': {'volume_id': '123456', 'force': False, 'name': None, 'description': None, 'metadata': {}}}
    self.assert_called_anytime('POST', '/snapshots', body=pre_3_66_request_body)