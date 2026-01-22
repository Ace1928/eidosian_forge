from openstack.compute.v2 import server
from openstack.tests.functional.compute import base as ft_base
from openstack.tests.functional.network.v2 import test_network
class TestServerAdmin(ft_base.BaseComputeTest):

    def setUp(self):
        super(TestServerAdmin, self).setUp()
        self._set_operator_cloud(interface='admin')
        self.NAME = 'needstobeshortandlowercase'
        self.USERDATA = 'SSdtIGFjdHVhbGx5IGEgZ29hdC4='
        volume = self.conn.create_volume(1)
        sot = self.conn.compute.create_server(name=self.NAME, flavor_id=self.flavor.id, image_id=self.image.id, networks='none', user_data=self.USERDATA, block_device_mapping=[{'uuid': volume.id, 'source_type': 'volume', 'boot_index': 0, 'destination_type': 'volume', 'delete_on_termination': True, 'volume_size': 1}])
        self.conn.compute.wait_for_server(sot, wait=self._wait_for_timeout)
        assert isinstance(sot, server.Server)
        self.assertEqual(self.NAME, sot.name)
        self.server = sot

    def tearDown(self):
        sot = self.conn.compute.delete_server(self.server.id)
        self.conn.compute.wait_for_delete(self.server, wait=self._wait_for_timeout)
        self.assertIsNone(sot)
        super(TestServerAdmin, self).tearDown()

    def test_get(self):
        sot = self.conn.compute.get_server(self.server.id)
        self.assertIsNotNone(sot.reservation_id)
        self.assertIsNotNone(sot.launch_index)
        self.assertIsNotNone(sot.ramdisk_id)
        self.assertIsNotNone(sot.kernel_id)
        self.assertEqual(self.NAME, sot.hostname)
        self.assertTrue(sot.root_device_name.startswith('/dev'))
        self.assertEqual(self.USERDATA, sot.user_data)
        self.assertTrue(sot.attached_volumes[0]['delete_on_termination'])