from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import transfer
from openstack import resource
from openstack.tests.unit import base
class TestTransfer(base.TestCase):

    def setUp(self):
        super().setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.resp.headers = {}
        self.resp.status_code = 202
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess.default_microversion = '3.55'

    def test_basic(self):
        tr = transfer.Transfer(TRANSFER)
        self.assertEqual('transfer', tr.resource_key)
        self.assertEqual('transfers', tr.resources_key)
        self.assertEqual('/volume-transfers', tr.base_path)
        self.assertTrue(tr.allow_create)
        self.assertIsNotNone(tr._max_microversion)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker'}, tr._query_mapping._mapping)

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
    @mock.patch.object(resource.Resource, '_translate_response')
    def test_create(self, mock_mv, mock_translate):
        sot = transfer.Transfer()
        sot.create(self.sess, volume_id=FAKE_VOL_ID, name=FAKE_VOL_NAME)
        self.sess.post.assert_called_with('/volume-transfers', json={'transfer': {}}, microversion='3.55', headers={}, params={'volume_id': FAKE_VOL_ID, 'name': FAKE_VOL_NAME})

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
    @mock.patch.object(resource.Resource, '_translate_response')
    def test_create_pre_v355(self, mock_mv, mock_translate):
        self.sess.default_microversion = '3.0'
        sot = transfer.Transfer()
        sot.create(self.sess, volume_id=FAKE_VOL_ID, name=FAKE_VOL_NAME)
        self.sess.post.assert_called_with('/os-volume-transfer', json={'transfer': {}}, microversion='3.0', headers={}, params={'volume_id': FAKE_VOL_ID, 'name': FAKE_VOL_NAME})

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
    @mock.patch.object(resource.Resource, '_translate_response')
    def test_accept(self, mock_mv, mock_translate):
        sot = transfer.Transfer()
        sot.id = FAKE_TRANSFER
        sot.accept(self.sess, auth_key=FAKE_AUTH_KEY)
        self.sess.post.assert_called_with('volume-transfers/%s/accept' % FAKE_TRANSFER, json={'accept': {'auth_key': FAKE_AUTH_KEY}}, microversion='3.55')

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
    @mock.patch.object(resource.Resource, '_translate_response')
    def test_accept_pre_v355(self, mock_mv, mock_translate):
        self.sess.default_microversion = '3.0'
        sot = transfer.Transfer()
        sot.id = FAKE_TRANSFER
        sot.accept(self.sess, auth_key=FAKE_AUTH_KEY)
        self.sess.post.assert_called_with('os-volume-transfer/%s/accept' % FAKE_TRANSFER, json={'accept': {'auth_key': FAKE_AUTH_KEY}}, microversion='3.0')