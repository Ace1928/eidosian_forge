from openstack import exceptions
from openstack.tests.functional import base
def test_update_network(self):
    net = self.operator_cloud.create_network(name=self.network_name)
    self.assertEqual(net.name, self.network_name)
    new_name = self.getUniqueString('network')
    net = self.operator_cloud.update_network(net.id, name=new_name)
    self.addCleanup(self.operator_cloud.delete_network, new_name)
    self.assertNotEqual(net.name, self.network_name)
    self.assertEqual(net.name, new_name)