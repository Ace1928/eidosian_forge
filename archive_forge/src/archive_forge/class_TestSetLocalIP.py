from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetLocalIP(TestLocalIP):
    _local_ip = network_fakes.create_one_local_ip()

    def setUp(self):
        super().setUp()
        self.network_client.update_local_ip = mock.Mock(return_value=None)
        self.network_client.find_local_ip = mock.Mock(return_value=self._local_ip)
        self.cmd = local_ip.SetLocalIP(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self._local_ip.name]
        verifylist = [('local_ip', self._local_ip.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_local_ip.assert_not_called()
        self.assertIsNone(result)

    def test_set_name_and_description(self):
        arglist = ['--name', 'new_local_ip_name', '--description', 'new_local_ip_description', self._local_ip.name]
        verifylist = [('name', 'new_local_ip_name'), ('description', 'new_local_ip_description'), ('local_ip', self._local_ip.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'new_local_ip_name', 'description': 'new_local_ip_description'}
        self.network_client.update_local_ip.assert_called_with(self._local_ip, **attrs)
        self.assertIsNone(result)