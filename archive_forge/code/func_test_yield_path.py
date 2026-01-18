import io
from unittest import mock
from glance_store.tests.unit.cinder import test_base as test_base_connector
def test_yield_path(self):
    fake_vol = mock.MagicMock(size=1)
    fake_device = io.BytesIO(b'fake binary data')
    fake_dev_path = self.connector.yield_path(fake_vol, fake_device)
    self.assertEqual(fake_device, fake_dev_path)