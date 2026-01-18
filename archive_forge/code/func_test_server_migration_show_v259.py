from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_server_migration_show_v259(self):
    self._set_mock_microversion('2.59')
    self.columns += ('UUID',)
    self.data += (self.server_migration.uuid,)
    self._test_server_migration_show()