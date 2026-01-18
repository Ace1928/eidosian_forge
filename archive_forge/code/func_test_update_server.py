import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_update_server(self):
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    self.user_cloud.create_server(name=self.server_name, image=self.image, flavor=self.flavor, wait=True)
    server_updated = self.user_cloud.update_server(self.server_name, name='new_name')
    self.assertEqual('new_name', server_updated['name'])