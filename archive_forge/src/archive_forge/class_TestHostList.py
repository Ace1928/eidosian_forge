from unittest import mock
from openstackclient.compute.v2 import host
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.host_list')
class TestHostList(compute_fakes.TestComputev2):
    _host = compute_fakes.create_one_host()

    def setUp(self):
        super(TestHostList, self).setUp()
        self.compute_sdk_client.get.return_value = fakes.FakeResponse(data={'hosts': [self._host]})
        self.columns = ('Host Name', 'Service', 'Zone')
        self.data = [(self._host['host_name'], self._host['service'], self._host['zone'])]
        self.cmd = host.ListHost(self.app, None)

    def test_host_list_no_option(self, h_mock):
        h_mock.return_value = [self._host]
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.get.assert_called_with('/os-hosts', microversion='2.1')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_host_list_with_option(self, h_mock):
        h_mock.return_value = [self._host]
        arglist = ['--zone', self._host['zone']]
        verifylist = [('zone', self._host['zone'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.get.assert_called_with('/os-hosts', microversion='2.1')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))