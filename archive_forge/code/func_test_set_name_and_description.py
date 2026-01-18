from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_name_and_description(self):
    arglist = ['--name', 'new_local_ip_name', '--description', 'new_local_ip_description', self._local_ip.name]
    verifylist = [('name', 'new_local_ip_name'), ('description', 'new_local_ip_description'), ('local_ip', self._local_ip.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'new_local_ip_name', 'description': 'new_local_ip_description'}
    self.network_client.update_local_ip.assert_called_with(self._local_ip, **attrs)
    self.assertIsNone(result)