import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class MetadefLoadUnloadTests:
    _namespace_count = 33
    _namespace_object_counts = {'OS::Compute::Quota': 3, 'OS::Software::WebServers': 3, 'OS::Software::DBMS': 12, 'OS::Software::Runtimes': 5}
    _namespace_property_counts = {'CIM::ProcessorAllocationSettingData': 3, 'CIM::ResourceAllocationSettingData': 19, 'CIM::StorageAllocationSettingData': 13, 'CIM::VirtualSystemSettingData': 17, 'OS::Compute::XenAPI': 1, 'OS::Compute::InstanceData': 2, 'OS::Compute::Libvirt': 4, 'OS::Compute::VMwareQuotaFlavor': 2, 'OS::Cinder::Volumetype': 1, 'OS::Glance::Signatures': 4, 'OS::Compute::AggregateIoOpsFilter': 1, 'OS::Compute::RandomNumberGenerator': 3, 'OS::Compute::VTPM': 2, 'OS::Compute::Hypervisor': 2, 'OS::Compute::CPUPinning': 2, 'OS::OperatingSystem': 3, 'OS::Compute::AggregateDiskFilter': 1, 'OS::Compute::AggregateNumInstancesFilter': 1, 'OS::Compute::CPUMode': 1, 'OS::Compute::HostCapabilities': 7, 'OS::Compute::VirtCPUTopology': 6, 'OS::Glance::CommonImageProperties': 10, 'OS::Compute::GuestShutdownBehavior': 1, 'OS::Compute::VMwareFlavor': 2, 'OS::Compute::TPM': 1, 'OS::Compute::GuestMemoryBacking': 1, 'OS::Compute::LibvirtImage': 16, 'OS::Compute::VMware': 6, 'OS::Compute::Watchdog': 1}

    def test_metadef_load_unload(self):
        metadata.db_load_metadefs(self.db_api.get_engine())
        expected = self._namespace_count
        namespaces = self.db_api.metadef_namespace_get_all(self.adm_context)
        actual = len(namespaces)
        self.assertEqual(expected, actual, f'expected {expected} namespaces but got {actual}')
        for namespace in namespaces:
            expected = self._namespace_object_counts.get(namespace['namespace'], 0)
            objects = self.db_api.metadef_object_get_all(self.adm_context, namespace['namespace'])
            actual = len(objects)
            self.assertEqual(expected, actual, f'expected {expected} objects in {namespace['namespace']} namespace but got {actual}: {', '.join((o['name'] for o in objects))}')
        for namespace in namespaces:
            expected = self._namespace_property_counts.get(namespace['namespace'], 0)
            properties = self.db_api.metadef_property_get_all(self.adm_context, namespace['namespace'])
            actual = len(properties)
            self.assertEqual(expected, actual, f'expected {expected} properties in {namespace['namespace']} namespace but got {actual}: {', '.join((p['name'] for p in properties))}')
        metadata.db_unload_metadefs(self.db_api.get_engine())