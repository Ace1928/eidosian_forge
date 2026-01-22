import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
class TestImageSnapshot(base.TestCase):

    def setUp(self):
        super(TestImageSnapshot, self).setUp()
        self.server_id = str(uuid.uuid4())
        self.image_id = str(uuid.uuid4())
        self.server_name = self.getUniqueString('name')
        self.fake_server = fakes.make_fake_server(self.server_id, self.server_name)

    def test_create_image_snapshot_wait_until_active_never_active(self):
        snapshot_name = 'test-snapshot'
        fake_image = fakes.make_fake_image(self.image_id, status='pending')
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri='{endpoint}/servers/{server_id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, server_id=self.server_id), headers=dict(Location='{endpoint}/images/{image_id}'.format(endpoint='https://images.example.com', image_id=self.image_id)), validate=dict(json={'createImage': {'name': snapshot_name, 'metadata': {}}})), self.get_glance_discovery_mock_dict(), dict(method='GET', uri='https://image.example.com/v2/images', json=dict(images=[fake_image]))])
        self.assertRaises(exceptions.ResourceTimeout, self.cloud.create_image_snapshot, snapshot_name, dict(id=self.server_id), wait=True, timeout=0.01)
        self.assert_calls(stop_after=5, do_count=False)

    def test_create_image_snapshot_wait_active(self):
        snapshot_name = 'test-snapshot'
        pending_image = fakes.make_fake_image(self.image_id, status='pending')
        fake_image = fakes.make_fake_image(self.image_id)
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri='{endpoint}/servers/{server_id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, server_id=self.server_id), headers=dict(Location='{endpoint}/images/{image_id}'.format(endpoint='https://images.example.com', image_id=self.image_id)), validate=dict(json={'createImage': {'name': snapshot_name, 'metadata': {}}})), self.get_glance_discovery_mock_dict(), dict(method='GET', uri='https://image.example.com/v2/images', json=dict(images=[pending_image])), dict(method='GET', uri='https://image.example.com/v2/images', json=dict(images=[fake_image]))])
        image = self.cloud.create_image_snapshot('test-snapshot', dict(id=self.server_id), wait=True, timeout=2)
        self.assertEqual(image['id'], self.image_id)
        self.assert_calls()