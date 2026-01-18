from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_list_default(self):
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_client.api.security_group_list.assert_called_once_with(search_opts={'all_tenants': False})
    self.assertEqual(self.expected_columns_no_group, columns)
    self.assertEqual(self.expected_data_no_group, list(data))