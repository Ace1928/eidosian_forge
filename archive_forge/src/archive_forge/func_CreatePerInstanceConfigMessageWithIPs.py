from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instance_groups.flags import AutoDeleteFlag
from googlecloudsdk.command_lib.compute.instance_groups.flags import STATEFUL_IP_DEFAULT_INTERFACE_NAME
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
def CreatePerInstanceConfigMessageWithIPs(holder, instance_ref, stateful_disks, stateful_metadata, stateful_internal_ips, stateful_external_ips, disk_getter=None):
    """Create per-instance config message from the given stateful attributes."""
    messages = holder.client.messages
    per_instance_config = CreatePerInstanceConfigMessage(holder, instance_ref, stateful_disks, stateful_metadata, disk_getter)
    preserved_state_internal_ips = []
    for stateful_internal_ip in stateful_internal_ips or []:
        preserved_state_internal_ips.append(MakePreservedStateInternalNetworkIpEntry(messages, stateful_internal_ip))
    per_instance_config.preservedState.internalIPs = messages.PreservedState.InternalIPsValue(additionalProperties=preserved_state_internal_ips)
    preserved_state_external_ips = []
    for stateful_external_ip in stateful_external_ips or []:
        preserved_state_external_ips.append(MakePreservedStateExternalNetworkIpEntry(messages, stateful_external_ip))
    per_instance_config.preservedState.externalIPs = messages.PreservedState.ExternalIPsValue(additionalProperties=preserved_state_external_ips)
    return per_instance_config