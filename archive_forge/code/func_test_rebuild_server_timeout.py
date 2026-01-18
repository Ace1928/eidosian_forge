import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_rebuild_server_timeout(self):
    """
        Test that a timeout while waiting for the server to rebuild raises an
        exception in rebuild_server.
        """
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id, 'action']), json={'server': self.rebuild_server}, validate=dict(json={'rebuild': {'imageRef': 'a'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id]), json={'server': self.rebuild_server})])
    self.assertRaises(exceptions.ResourceTimeout, self.cloud.rebuild_server, self.fake_server['id'], 'a', wait=True, timeout=0.001)
    self.assert_calls(do_count=False)