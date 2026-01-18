import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_rebuild_server_rebuild_exception(self):
    """
        Test that an exception in the rebuild raises an exception in
        rebuild_server.
        """
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id, 'action']), status_code=400, validate=dict(json={'rebuild': {'imageRef': 'a', 'adminPass': 'b'}}))])
    self.assertRaises(exceptions.SDKException, self.cloud.rebuild_server, self.fake_server['id'], 'a', 'b')
    self.assert_calls()