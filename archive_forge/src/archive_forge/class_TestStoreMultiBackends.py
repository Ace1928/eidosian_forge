from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
class TestStoreMultiBackends(utils.BaseTestCase):

    def setUp(self):
        self.store_api = unit_test_utils.FakeStoreAPI()
        self.store_utils = unit_test_utils.FakeStoreUtils(self.store_api)
        self.enabled_backends = {'ceph1': 'rbd', 'ceph2': 'rbd'}
        super(TestStoreMultiBackends, self).setUp()
        self.config(enabled_backends=self.enabled_backends)

    @mock.patch('glance.location.signature_utils.get_verifier')
    def test_set_data_calls_upload_to_store(self, msig):
        context = glance.context.RequestContext(user=USER1)
        extra_properties = {'img_signature_certificate_uuid': 'UUID', 'img_signature_hash_method': 'METHOD', 'img_signature_key_type': 'TYPE', 'img_signature': 'VALID'}
        image_stub = ImageStub(UUID2, status='queued', locations=[], extra_properties=extra_properties)
        image_stub.disk_format = 'iso'
        image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
        with mock.patch.object(image, '_upload_to_store') as mloc:
            image.set_data('YYYY', 4, backend='ceph1')
            msig.assert_called_once_with(context=context, img_signature_certificate_uuid='UUID', img_signature_hash_method='METHOD', img_signature='VALID', img_signature_key_type='TYPE')
            mloc.assert_called_once_with('YYYY', msig.return_value, 'ceph1', 4)
        self.assertEqual('active', image.status)

    def test_image_set_data(self):
        store_api = mock.MagicMock()
        store_api.add_with_multihash.return_value = ('rbd://ceph1', 4, 'Z', 'MH', {'backend': 'ceph1'})
        context = glance.context.RequestContext(user=USER1)
        image_stub = ImageStub(UUID2, status='queued', locations=[])
        image = glance.location.ImageProxy(image_stub, context, store_api, self.store_utils)
        image.set_data('YYYY', 4, backend='ceph1')
        self.assertEqual(4, image.size)
        self.assertEqual('rbd://ceph1', image.locations[0]['url'])
        self.assertEqual({'backend': 'ceph1'}, image.locations[0]['metadata'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)

    @mock.patch('glance.location.LOG')
    def test_image_set_data_valid_signature(self, mock_log):
        store_api = mock.MagicMock()
        store_api.add_with_multihash.return_value = ('rbd://ceph1', 4, 'Z', 'MH', {'backend': 'ceph1'})
        context = glance.context.RequestContext(user=USER1)
        extra_properties = {'img_signature_certificate_uuid': 'UUID', 'img_signature_hash_method': 'METHOD', 'img_signature_key_type': 'TYPE', 'img_signature': 'VALID'}
        image_stub = ImageStub(UUID2, status='queued', extra_properties=extra_properties)
        self.mock_object(signature_utils, 'get_verifier', unit_test_utils.fake_get_verifier)
        image = glance.location.ImageProxy(image_stub, context, store_api, self.store_utils)
        image.set_data('YYYY', 4, backend='ceph1')
        self.assertEqual('active', image.status)
        call = mock.call('Successfully verified signature for image %s', UUID2)
        mock_log.info.assert_has_calls([call])

    @mock.patch('glance.location.signature_utils.get_verifier')
    def test_image_set_data_invalid_signature(self, msig):
        msig.return_value.verify.side_effect = crypto_exception.InvalidSignature
        store_api = mock.MagicMock()
        store_api.add_with_multihash.return_value = ('rbd://ceph1', 4, 'Z', 'MH', {'backend': 'ceph1'})
        context = glance.context.RequestContext(user=USER1)
        extra_properties = {'img_signature_certificate_uuid': 'UUID', 'img_signature_hash_method': 'METHOD', 'img_signature_key_type': 'TYPE', 'img_signature': 'INVALID'}
        image_stub = ImageStub(UUID2, status='queued', extra_properties=extra_properties)
        image = glance.location.ImageProxy(image_stub, context, store_api, self.store_utils)
        self.assertRaises(cursive_exception.SignatureVerificationError, image.set_data, 'YYYY', 4, backend='ceph1')