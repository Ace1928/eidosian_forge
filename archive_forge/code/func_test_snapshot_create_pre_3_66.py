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
@ddt.data(True, False, 'Nonboolean')
@mock.patch('cinderclient.utils.find_resource')
def test_snapshot_create_pre_3_66(self, force_value, mock_find_vol):
    mock_find_vol.return_value = volumes.Volume(self, {'id': '123456'}, loaded=True)
    snap_body_3_65 = {'snapshot': {'volume_id': '123456', 'force': f'{force_value}', 'name': None, 'description': None, 'metadata': {}}}
    self.run_command(f'--os-volume-api-version 3.65 snapshot-create --force {force_value} 123456')
    self.assert_called_anytime('POST', '/snapshots', body=snap_body_3_65)