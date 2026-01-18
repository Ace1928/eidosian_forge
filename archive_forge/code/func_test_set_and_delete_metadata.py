import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_set_and_delete_metadata(self):
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    self.user_cloud.create_server(name=self.server_name, image=self.image, flavor=self.flavor, wait=True)
    self.user_cloud.set_server_metadata(self.server_name, {'key1': 'value1', 'key2': 'value2'})
    updated_server = self.user_cloud.get_server(self.server_name)
    self.assertEqual(set(updated_server.metadata.items()), set({'key1': 'value1', 'key2': 'value2'}.items()))
    self.user_cloud.set_server_metadata(self.server_name, {'key2': 'value3'})
    updated_server = self.user_cloud.get_server(self.server_name)
    self.assertEqual(set(updated_server.metadata.items()), set({'key1': 'value1', 'key2': 'value3'}.items()))
    self.user_cloud.delete_server_metadata(self.server_name, ['key2'])
    updated_server = self.user_cloud.get_server(self.server_name)
    self.assertEqual(set(updated_server.metadata.items()), set({'key1': 'value1'}.items()))
    self.user_cloud.delete_server_metadata(self.server_name, ['key1'])
    updated_server = self.user_cloud.get_server(self.server_name)
    self.assertEqual(set(updated_server.metadata.items()), set([]))
    self.assertRaises(exceptions.NotFoundException, self.user_cloud.delete_server_metadata, self.server_name, ['key1'])