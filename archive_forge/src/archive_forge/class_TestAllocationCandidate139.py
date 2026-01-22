import uuid
from osc_placement.tests.functional import base
class TestAllocationCandidate139(base.BaseTestCase):
    VERSION = '1.39'

    def setUp(self):
        super(TestAllocationCandidate139, self).setUp()
        self.rp1 = self.resource_provider_create()
        self.rp1_1 = self.resource_provider_create(parent_provider_uuid=self.rp1['uuid'])
        self.rp1_2 = self.resource_provider_create(parent_provider_uuid=self.rp1['uuid'])
        self.resource_inventory_set(self.rp1['uuid'], 'DISK_GB=512')
        self.resource_inventory_set(self.rp1_1['uuid'], 'VCPU=8', 'MEMORY_MB=8192')
        self.resource_inventory_set(self.rp1_2['uuid'], 'VCPU=16', 'MEMORY_MB=8192')
        self.resource_provider_trait_set(self.rp1['uuid'], 'STORAGE_DISK_HDD')
        self.resource_provider_trait_set(self.rp1_1['uuid'], 'HW_CPU_X86_AVX')
        self.resource_provider_trait_set(self.rp1_2['uuid'], 'HW_CPU_X86_SSE')
        self.rp2 = self.resource_provider_create()
        self.rp2_1 = self.resource_provider_create(parent_provider_uuid=self.rp2['uuid'])
        self.rp2_2 = self.resource_provider_create(parent_provider_uuid=self.rp2['uuid'])
        self.resource_inventory_set(self.rp2['uuid'], 'DISK_GB=512')
        self.resource_inventory_set(self.rp2_1['uuid'], 'VCPU=8', 'MEMORY_MB=8192')
        self.resource_inventory_set(self.rp2_2['uuid'], 'VCPU=16', 'MEMORY_MB=8192')
        self.resource_provider_trait_set(self.rp2['uuid'], 'STORAGE_DISK_SSD')
        self.resource_provider_trait_set(self.rp2_1['uuid'], 'HW_CPU_X86_AVX')
        self.resource_provider_trait_set(self.rp2_2['uuid'], 'HW_CPU_X86_SSE')

    def test_list_with_any_traits(self):
        groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX',)}}
        rows = self.allocation_candidate_granular(groups=groups)
        numbers = {row['#'] for row in rows}
        self.assertEqual(1, len(numbers))
        self.assertEqual(2, len(rows))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual({self.rp1['uuid'], self.rp1_1['uuid']}, rps)
        groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD,STORAGE_DISK_SSD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX',)}}
        rows = self.allocation_candidate_granular(groups=groups)
        numbers = {row['#'] for row in rows}
        self.assertEqual(2, len(numbers))
        self.assertEqual(4, len(rows))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual({self.rp1['uuid'], self.rp1_1['uuid'], self.rp2['uuid'], self.rp2_1['uuid']}, rps)
        groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD,STORAGE_DISK_SSD', 'STORAGE_DISK_SSD'), 'forbidden': ('STORAGE_DISK_HDD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX,HW_CPU_X86_SSE',), 'forbidden': ('HW_CPU_X86_SSE',)}}
        rows = self.allocation_candidate_granular(groups=groups)
        numbers = {row['#'] for row in rows}
        self.assertEqual(1, len(numbers))
        self.assertEqual(2, len(rows))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual({self.rp2['uuid'], self.rp2_1['uuid']}, rps)