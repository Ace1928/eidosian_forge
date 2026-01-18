from glance_store import capabilities as caps
from glance_store.tests import base
def test_store_capabilities_property(self):
    store1 = FakeStoreWithDynamicCapabilities()
    self.assertTrue(hasattr(store1, 'capabilities'))
    store2 = FakeStoreWithMixedCapabilities()
    self.assertEqual(store1.capabilities, store2.capabilities)