from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeSpecificSKUReservationMessage(messages, vm_count, accelerators, local_ssds, machine_type, min_cpu_platform, location_hint=None, freeze_duration=None, freeze_interval=None, source_instance_template_ref=None):
    """Constructs a single specific sku reservation message object."""
    prop_msgs = messages.AllocationSpecificSKUAllocationReservedInstanceProperties
    if source_instance_template_ref:
        return messages.AllocationSpecificSKUReservation(count=vm_count, sourceInstanceTemplate=source_instance_template_ref.SelfLink(), instanceProperties=None)
    else:
        instance_properties = prop_msgs(guestAccelerators=accelerators, localSsds=local_ssds, machineType=machine_type, minCpuPlatform=min_cpu_platform)
        if freeze_duration:
            instance_properties.maintenanceFreezeDurationHours = freeze_duration // 3600
        if freeze_interval:
            instance_properties.maintenanceInterval = messages.AllocationSpecificSKUAllocationReservedInstanceProperties.MaintenanceIntervalValueValuesEnum(freeze_interval)
        if location_hint:
            instance_properties.locationHint = location_hint
        return messages.AllocationSpecificSKUReservation(count=vm_count, instanceProperties=instance_properties)