import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_rebuild_server_no_wait(self):
    """
        Test that rebuild_server with no wait and no exception in the
        rebuild call returns the server instance.
        """
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id, 'action']), json={'server': self.rebuild_server}, validate=dict(json={'rebuild': {'imageRef': 'a'}})), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []})])
    self.assertEqual(self.rebuild_server['status'], self.cloud.rebuild_server(self.fake_server['id'], 'a')['status'])
    self.assert_calls()