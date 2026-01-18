import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_rebuild_server_wait(self):
    """
        Test that rebuild_server with a wait returns the server instance when
        its status changes to "ACTIVE".
        """
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id, 'action']), json={'server': self.rebuild_server}, validate=dict(json={'rebuild': {'imageRef': 'a'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id]), json={'server': self.rebuild_server}), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id]), json={'server': self.fake_server}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []})])
    self.assertEqual('ACTIVE', self.cloud.rebuild_server(self.fake_server['id'], 'a', wait=True)['status'])
    self.assert_calls()