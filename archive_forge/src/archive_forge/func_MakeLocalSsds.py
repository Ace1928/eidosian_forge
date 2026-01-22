from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeLocalSsds(messages, ssd_configs):
    """Constructs the repeated local_ssd message objects."""
    if ssd_configs is None:
        return []
    local_ssds = []
    disk_msg = messages.AllocationSpecificSKUAllocationAllocatedInstancePropertiesReservedDisk
    interface_msg = disk_msg.InterfaceValueValuesEnum
    total_partitions = 0
    for s in ssd_configs:
        if s['interface'].upper() == 'NVME':
            interface = interface_msg.NVME
        else:
            interface = interface_msg.SCSI
        m = disk_msg(diskSizeGb=s['size'], interface=interface)
        partitions = s.get('count', 1)
        if partitions not in range(24 + 1):
            raise exceptions.InvalidArgumentException('--local-ssd', 'The number of SSDs attached to an instance must be in the range of 1-24.')
        total_partitions += partitions
        if total_partitions > 24:
            raise exceptions.InvalidArgumentException('--local-ssd', 'The total number of SSDs attached to an instance must not exceed 24.')
        local_ssds.extend([m] * partitions)
    return local_ssds