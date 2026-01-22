from oslo_serialization import jsonutils
from novaclient.tests.functional.v2.legacy import test_extended_attributes
class TestExtAttrNovaClientV23(test_extended_attributes.TestExtAttrNovaClient):
    """Functional tests for os-extended-server-attributes, microversion 2.3"""
    COMPUTE_API_VERSION = '2.3'

    def test_extended_server_attributes(self):
        server, volume = self._create_server_and_attach_volume()
        table = self.nova('show %s' % server.id)
        for attr in ['OS-EXT-SRV-ATTR:reservation_id', 'OS-EXT-SRV-ATTR:launch_index', 'OS-EXT-SRV-ATTR:ramdisk_id', 'OS-EXT-SRV-ATTR:kernel_id', 'OS-EXT-SRV-ATTR:hostname', 'OS-EXT-SRV-ATTR:root_device_name']:
            self._get_value_from_the_table(table, attr)
        volume_attr = self._get_value_from_the_table(table, 'os-extended-volumes:volumes_attached')
        self.assertIn('delete_on_termination', jsonutils.loads(volume_attr)[0])