from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_list_with_group(self):
    arglist = [self._security_group['id']]
    verifylist = [('group', self._security_group['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_client.api.security_group_find.assert_called_once_with(self._security_group['id'])
    self.assertEqual(self.expected_columns_with_group, columns)
    self.assertEqual(self.expected_data_with_group, list(data))