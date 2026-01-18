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
def test_restore_with_az(self):
    self.run_command('--os-volume-api-version 3.47 backup-restore 1234 --name restore_vol --availability-zone restore_az')
    expected = {'volume': {'size': 10, 'name': 'restore_vol', 'availability_zone': 'restore_az', 'backup_id': '1234', 'metadata': {}, 'imageRef': None, 'source_volid': None, 'consistencygroup_id': None, 'snapshot_id': None, 'volume_type': None, 'description': None}}
    self.assert_called('POST', '/volumes', body=expected)