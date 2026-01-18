from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_group_fields(self):
    self.create_node()
    self.create_port_group(address='11:22:33:44:55:66')
    result = self.conn.baremetal.port_groups(fields=['uuid', 'name'])
    for item in result:
        self.assertIsNotNone(item.id)
        self.assertIsNone(item.address)