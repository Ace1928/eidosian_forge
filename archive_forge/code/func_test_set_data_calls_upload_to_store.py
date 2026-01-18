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