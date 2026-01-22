from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def _setup_security_group_rule(self, attrs=None):
    self._security_group_rule = network_fakes.FakeSecurityGroupRule.create_one_security_group_rule(attrs)
    self.network_client.create_security_group_rule = mock.Mock(return_value=self._security_group_rule)
    self.expected_data = (self._security_group_rule.description, self._security_group_rule.direction, self._security_group_rule.ether_type, self._security_group_rule.id, self._security_group_rule.port_range_max, self._security_group_rule.port_range_min, self._security_group_rule.project_id, self._security_group_rule.protocol, self._security_group_rule.remote_address_group_id, self._security_group_rule.remote_group_id, self._security_group_rule.remote_ip_prefix, self._security_group_rule.security_group_id)