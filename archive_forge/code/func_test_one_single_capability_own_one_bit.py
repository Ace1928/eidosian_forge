from glance_store import capabilities as caps
from glance_store.tests import base
def test_one_single_capability_own_one_bit(self):
    cap_list = [caps.BitMasks.READ_ACCESS, caps.BitMasks.WRITE_ACCESS, caps.BitMasks.DRIVER_REUSABLE]
    for cap in cap_list:
        self.assertEqual(1, bin(cap).count('1'))