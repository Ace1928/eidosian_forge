from unittest import mock
from unittest.mock import call
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import service
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_service_set_find_service_by_host_and_binary_many_results(self):
    self.compute_sdk_client.services.return_value = [mock.Mock(), mock.Mock()]
    ex = self.assertRaises(exceptions.CommandError, self.cmd._find_service_by_host_and_binary, self.compute_sdk_client, 'fake-host', 'nova-compute')
    self.assertIn('Multiple compute services found for host "fake-host" and binary "nova-compute". Unable to proceed.', str(ex))