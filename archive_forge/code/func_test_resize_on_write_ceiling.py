import hashlib
import io
from unittest import mock
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils as test_utils
def test_resize_on_write_ceiling(self):
    image = mock.MagicMock()
    ret = self.store._resize_on_write(image, 32, 16, 16)
    self.assertEqual(0, ret)
    image.resize.assert_not_called()
    self.store.size = 8
    ret = self.store._resize_on_write(image, 0, 16, 16)
    self.assertEqual(8 + self.store.WRITE_CHUNKSIZE * 2, ret)
    self.assertEqual(self.store.WRITE_CHUNKSIZE * 2, self.store.resize_amount)
    image.resize.assert_called_once_with(ret)
    image.resize.reset_mock()
    self.store.size = ret
    ret = self.store._resize_on_write(image, 0, 64, 16)
    self.assertEqual(8 + self.store.WRITE_CHUNKSIZE * 2, ret)
    image.resize.assert_not_called()
    ret = self.store._resize_on_write(image, 0, ret + 1, 16)
    self.assertEqual(8 + self.store.WRITE_CHUNKSIZE * 6, ret)
    image.resize.assert_called_once_with(ret)
    self.assertEqual(self.store.WRITE_CHUNKSIZE * 4, self.store.resize_amount)
    image.resize.reset_mock()
    self.store.resize_amount = 2 * units.Gi
    self.store.size = 1 * units.Gi
    ret = self.store._resize_on_write(image, 0, 4097 * units.Mi, 16)
    self.assertEqual(4 * units.Gi, self.store.resize_amount)
    self.assertEqual((1 + 4) * units.Gi, ret)
    self.store.size = ret
    ret = self.store._resize_on_write(image, 0, 6144 * units.Mi, 16)
    self.assertEqual(8 * units.Gi, self.store.resize_amount)
    self.assertEqual((1 + 4 + 8) * units.Gi, ret)
    self.store.size = ret
    ret = self.store._resize_on_write(image, 0, 14336 * units.Mi, 16)
    self.assertEqual(8 * units.Gi, self.store.resize_amount)
    self.assertEqual((1 + 4 + 8 + 8) * units.Gi, ret)
    self.store.size = ret
    ret = self.store._resize_on_write(image, 0, 22528 * units.Mi, 16)
    self.assertEqual(8 * units.Gi, self.store.resize_amount)
    self.assertEqual((1 + 4 + 8 + 8 + 8) * units.Gi, ret)
    image.resize.assert_has_calls([mock.call(5 * units.Gi), mock.call(13 * units.Gi), mock.call(21 * units.Gi), mock.call(29 * units.Gi)])