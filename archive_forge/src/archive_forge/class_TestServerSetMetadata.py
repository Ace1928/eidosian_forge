import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
class TestServerSetMetadata(base.TestCase):

    def setUp(self):
        super(TestServerSetMetadata, self).setUp()
        self.server_id = str(uuid.uuid4())
        self.server_name = self.getUniqueString('name')
        self.fake_server = fakes.make_fake_server(self.server_id, self.server_name)

    def test_server_set_metadata_with_exception(self):
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [self.fake_server]}), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', self.fake_server['id'], 'metadata']), validate=dict(json={'metadata': {'meta': 'data'}}), json={}, status_code=400)])
        self.assertRaises(exceptions.BadRequestException, self.cloud.set_server_metadata, self.server_name, {'meta': 'data'})
        self.assert_calls()

    def test_server_set_metadata(self):
        metadata = {'meta': 'data'}
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [self.fake_server]}), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', self.fake_server['id'], 'metadata']), validate=dict(json={'metadata': metadata}), status_code=200, json={'metadata': metadata})])
        self.cloud.set_server_metadata(self.server_id, metadata)
        self.assert_calls()