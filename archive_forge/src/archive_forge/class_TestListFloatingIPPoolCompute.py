from unittest import mock
from openstackclient.network.v2 import floating_ip_pool
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch('openstackclient.api.compute_v2.APIv2.floating_ip_pool_list')
class TestListFloatingIPPoolCompute(compute_fakes.TestComputev2):
    _floating_ip_pools = compute_fakes.create_floating_ip_pools(count=3)
    columns = ('Name',)
    data = []
    for pool in _floating_ip_pools:
        data.append((pool['name'],))

    def setUp(self):
        super(TestListFloatingIPPoolCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = floating_ip_pool.ListFloatingIPPool(self.app, None)

    def test_floating_ip_list(self, fipp_mock):
        fipp_mock.return_value = self._floating_ip_pools
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        fipp_mock.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))