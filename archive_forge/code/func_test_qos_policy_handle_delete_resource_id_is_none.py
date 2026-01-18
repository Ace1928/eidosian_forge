from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_policy_handle_delete_resource_id_is_none(self):
    self.my_qos_policy.resource_id = None
    self.assertIsNone(self.my_qos_policy.handle_delete())
    self.assertEqual(0, self.neutronclient.delete_qos_policy.call_count)