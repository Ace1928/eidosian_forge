from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_set_all_options(self, sg_mock):
    sg_mock.return_value = mock.Mock(return_value=None)
    new_name = 'new-' + self._security_group['name']
    new_description = 'new-' + self._security_group['description']
    arglist = ['--name', new_name, '--description', new_description, self._security_group['name']]
    verifylist = [('description', new_description), ('group', self._security_group['name']), ('name', new_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    sg_mock.assert_called_once_with(self._security_group, new_name, new_description)
    self.assertIsNone(result)