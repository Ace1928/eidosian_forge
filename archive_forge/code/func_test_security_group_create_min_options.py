from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_create_min_options(self, sg_mock):
    sg_mock.return_value = self._security_group
    arglist = [self._security_group['name']]
    verifylist = [('name', self._security_group['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    sg_mock.assert_called_once_with(self._security_group['name'], self._security_group['name'])
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)