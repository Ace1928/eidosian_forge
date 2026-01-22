from oslo_serialization import jsonutils
from novaclient.tests.functional import base
class TestExtAttrNovaClient(base.ClientTestBase):
    """Functional tests for os-extended-server-attributes"""
    COMPUTE_API_VERSION = '2.1'

    def _create_server_and_attach_volume(self):
        server = self._create_server()
        volume = self.cinder.volumes.create(1)
        self.addCleanup(volume.delete)
        self.wait_for_volume_status(volume, 'available')
        self.nova('volume-attach', params='%s %s' % (server.name, volume.id))
        self.addCleanup(self._release_volume, server, volume)
        self.wait_for_volume_status(volume, 'in-use')
        return (server, volume)

    def _release_volume(self, server, volume):
        self.nova('volume-detach', params='%s %s' % (server.id, volume.id))
        self.wait_for_volume_status(volume, 'available')

    def test_extended_server_attributes(self):
        server, volume = self._create_server_and_attach_volume()
        table = self.nova('show %s' % server.id)
        for attr in ['OS-EXT-SRV-ATTR:host', 'OS-EXT-SRV-ATTR:hypervisor_hostname', 'OS-EXT-SRV-ATTR:instance_name']:
            self._get_value_from_the_table(table, attr)
        volume_attr = self._get_value_from_the_table(table, 'os-extended-volumes:volumes_attached')
        self.assertIn('id', jsonutils.loads(volume_attr)[0])