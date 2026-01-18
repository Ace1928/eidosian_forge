from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_negative_non_existing(self):
    uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_port, uuid)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.find_port, uuid, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.delete_port, uuid, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.update_port, uuid, pxe_enabled=True)
    self.assertIsNone(self.conn.baremetal.find_port(uuid))
    self.assertIsNone(self.conn.baremetal.delete_port(uuid))