from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_policy_handle_create(self):
    policy = {'policy': {'description': 'a policy for test', 'id': '9c1eb3fe-7bba-479d-bd43-1d497e53c384', 'rules': [], 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0', 'shared': True}}
    create_props = {'name': 'test_policy', 'description': 'a policy for test', 'shared': True, 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}
    self.neutronclient.create_qos_policy.return_value = policy
    self.my_qos_policy.handle_create()
    self.assertEqual('9c1eb3fe-7bba-479d-bd43-1d497e53c384', self.my_qos_policy.resource_id)
    self.neutronclient.create_qos_policy.assert_called_once_with({'policy': create_props})