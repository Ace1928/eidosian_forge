import operator
import uuid
from osc_placement.tests.functional import base
def test_show_required_trait(self):
    rp1 = self.resource_provider_create()
    rp2 = self.resource_provider_create()
    self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
    self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
    self.resource_provider_trait_set(rp1['uuid'], 'STORAGE_DISK_SSD', 'HW_NIC_SRIOV_MULTIQUEUE')
    self.resource_provider_trait_set(rp2['uuid'], 'STORAGE_DISK_HDD', 'HW_NIC_SRIOV_MULTIQUEUE')
    rps = self.resource_provider_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('HW_NIC_SRIOV_MULTIQUEUE',))
    uuids = [rp['uuid'] for rp in rps]
    self.assertEqual(2, len(rps))
    self.assertIn(rp1['uuid'], uuids)
    self.assertIn(rp2['uuid'], uuids)
    rps = self.resource_provider_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('STORAGE_DISK_HDD', 'HW_NIC_SRIOV_MULTIQUEUE'))
    self.assertEqual(1, len(rps))
    self.assertEqual(rp2['uuid'], rps[0]['uuid'])