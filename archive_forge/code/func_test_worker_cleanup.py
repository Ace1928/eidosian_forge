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
def test_worker_cleanup(self):
    self.run_command('--os-volume-api-version 3.24 work-cleanup --cluster clustername --host hostname --binary binaryname --is-up false --disabled true --resource-id uuid --resource-type Volume --service-id 1')
    expected = {'cluster_name': 'clustername', 'host': 'hostname', 'binary': 'binaryname', 'is_up': 'false', 'disabled': 'true', 'resource_id': 'uuid', 'resource_type': 'Volume', 'service_id': 1}
    self.assert_called('POST', '/workers/cleanup', body=expected)