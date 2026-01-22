from glance_store import capabilities as caps
from glance_store.tests import base
class TestStoreCapabilities(base.StoreBaseTest):

    def _verify_store_capabilities(self, store):
        self.assertTrue(store.is_capable(caps.BitMasks.READ_RANDOM))
        self.assertTrue(store.is_capable(caps.BitMasks.DRIVER_REUSABLE))
        self.assertFalse(store.is_capable(caps.BitMasks.WRITE_ACCESS))

    def test_static_capabilities_setup(self):
        self._verify_store_capabilities(FakeStoreWithStaticCapabilities())

    def test_dynamic_capabilities_setup(self):
        self._verify_store_capabilities(FakeStoreWithDynamicCapabilities())

    def test_mixed_capabilities_setup(self):
        self._verify_store_capabilities(FakeStoreWithMixedCapabilities())

    def test_set_unset_capabilities(self):
        store = FakeStoreWithStaticCapabilities()
        self.assertFalse(store.is_capable(caps.BitMasks.WRITE_ACCESS))
        store.set_capabilities(caps.BitMasks.WRITE_ACCESS)
        self.assertTrue(store.is_capable(caps.BitMasks.WRITE_ACCESS))
        store.unset_capabilities(caps.BitMasks.WRITE_ACCESS)
        self.assertFalse(store.is_capable(caps.BitMasks.WRITE_ACCESS))
        cap_list = [caps.BitMasks.WRITE_ACCESS, caps.BitMasks.WRITE_OFFSET]
        store.set_capabilities(*cap_list)
        self.assertTrue(store.is_capable(*cap_list))
        store.unset_capabilities(*cap_list)
        self.assertFalse(store.is_capable(*cap_list))

    def test_store_capabilities_property(self):
        store1 = FakeStoreWithDynamicCapabilities()
        self.assertTrue(hasattr(store1, 'capabilities'))
        store2 = FakeStoreWithMixedCapabilities()
        self.assertEqual(store1.capabilities, store2.capabilities)

    def test_cascaded_unset_capabilities(self):
        store = FakeStoreWithMixedCapabilities()
        self._verify_store_capabilities(store)
        store.unset_capabilities(caps.BitMasks.READ_ACCESS)
        cap_list = [caps.BitMasks.READ_ACCESS, caps.BitMasks.READ_OFFSET, caps.BitMasks.READ_CHUNK, caps.BitMasks.READ_RANDOM]
        for cap in cap_list:
            self.assertFalse(store.is_capable(cap))
        self.assertTrue(store.is_capable(caps.BitMasks.DRIVER_REUSABLE))
        store = FakeStoreWithDynamicCapabilities(caps.BitMasks.WRITE_RANDOM, caps.BitMasks.DRIVER_REUSABLE)
        self.assertTrue(store.is_capable(caps.BitMasks.WRITE_RANDOM))
        self.assertTrue(store.is_capable(caps.BitMasks.DRIVER_REUSABLE))
        store.unset_capabilities(caps.BitMasks.WRITE_ACCESS)
        cap_list = [caps.BitMasks.WRITE_ACCESS, caps.BitMasks.WRITE_OFFSET, caps.BitMasks.WRITE_CHUNK, caps.BitMasks.WRITE_RANDOM]
        for cap in cap_list:
            self.assertFalse(store.is_capable(cap))
        self.assertTrue(store.is_capable(caps.BitMasks.DRIVER_REUSABLE))