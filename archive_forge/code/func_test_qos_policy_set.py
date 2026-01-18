import uuid
from openstackclient.tests.functional.network.v2 import common
def test_qos_policy_set(self):
    policy_name = uuid.uuid4().hex
    json_output = self.openstack('network qos policy create ' + policy_name, parse_output=True)
    self.addCleanup(self.openstack, 'network qos policy delete ' + policy_name)
    self.assertEqual(policy_name, json_output['name'])
    self.openstack('network qos policy set ' + '--share ' + policy_name)
    json_output = self.openstack('network qos policy show ' + policy_name, parse_output=True)
    self.assertTrue(json_output['shared'])
    self.openstack('network qos policy set ' + '--no-share ' + '--no-default ' + policy_name)
    json_output = self.openstack('network qos policy show ' + policy_name, parse_output=True)
    self.assertFalse(json_output['shared'])
    self.assertFalse(json_output['is_default'])