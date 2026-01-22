from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteLocalIP(TestLocalIP):
    _local_ips = network_fakes.create_local_ips(count=2)

    def setUp(self):
        super().setUp()
        self.network_client.delete_local_ip = mock.Mock(return_value=None)
        self.network_client.find_local_ip = network_fakes.get_local_ips(local_ips=self._local_ips)
        self.cmd = local_ip.DeleteLocalIP(self.app, self.namespace)

    def test_local_ip_delete(self):
        arglist = [self._local_ips[0].name]
        verifylist = [('local_ip', [self._local_ips[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_local_ip.assert_called_once_with(self._local_ips[0].name, ignore_missing=False)
        self.network_client.delete_local_ip.assert_called_once_with(self._local_ips[0])
        self.assertIsNone(result)

    def test_multi_local_ips_delete(self):
        arglist = []
        for a in self._local_ips:
            arglist.append(a.name)
        verifylist = [('local_ip', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for a in self._local_ips:
            calls.append(call(a))
        self.network_client.delete_local_ip.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_local_ips_delete_with_exception(self):
        arglist = [self._local_ips[0].name, 'unexist_local_ip']
        verifylist = [('local_ip', [self._local_ips[0].name, 'unexist_local_ip'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._local_ips[0], exceptions.CommandError]
        self.network_client.find_local_ip = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 local IPs failed to delete.', str(e))
        self.network_client.find_local_ip.assert_any_call(self._local_ips[0].name, ignore_missing=False)
        self.network_client.find_local_ip.assert_any_call('unexist_local_ip', ignore_missing=False)
        self.network_client.delete_local_ip.assert_called_once_with(self._local_ips[0])