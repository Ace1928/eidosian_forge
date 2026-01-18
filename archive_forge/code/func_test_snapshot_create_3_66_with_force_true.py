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
@ddt.data(True, 'true', 'on', '1')
@mock.patch('cinderclient.utils.find_resource')
def test_snapshot_create_3_66_with_force_true(self, f_val, mock_find_vol):
    mock_find_vol.return_value = volumes.Volume(self, {'id': '123456'}, loaded=True)
    mock_find_vol.return_value = volumes.Volume(self, {'id': '123456'}, loaded=True)
    self.run_command(f'--os-volume-api-version 3.66 snapshot-create --force {f_val} 123456')
    self.assert_called_anytime('POST', '/snapshots', body=self.SNAP_BODY_3_66)