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
def _add_image(self, context, image_id, data, len):
    image_stub = ImageStub(image_id, status='queued', locations=[])
    image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
    image.set_data(data, len)
    self.assertEqual(len, image.size)
    location = {'url': image_id, 'metadata': {}, 'status': 'active'}
    self.assertEqual([location], image.locations)
    self.assertEqual([location], image_stub.locations)
    self.assertEqual('active', image.status)
    return (image, image_stub)