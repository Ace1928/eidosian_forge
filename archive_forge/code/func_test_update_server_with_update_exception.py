import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_server_with_update_exception(self):
    """
        Test that an exception in the update raises an exception in
        update_server.
        """
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=%s' % self.server_name]), json={'servers': [self.fake_server]}), dict(method='PUT', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id]), status_code=400, validate=dict(json={'server': {'name': self.updated_server_name}}))])
    self.assertRaises(exceptions.SDKException, self.cloud.update_server, self.server_name, name=self.updated_server_name)
    self.assert_calls()