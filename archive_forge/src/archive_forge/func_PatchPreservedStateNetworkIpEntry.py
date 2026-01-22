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
def PatchPreservedStateNetworkIpEntry(messages, stateful_ip_to_patch, update_stateful_ip):
    """Prepares stateful ip preserved state entry."""
    auto_delete = update_stateful_ip.get('auto-delete')
    if auto_delete:
        stateful_ip_to_patch.autoDelete = auto_delete.GetAutoDeleteEnumValue(messages.PreservedStatePreservedNetworkIp.AutoDeleteValueValuesEnum)
    ip_address = update_stateful_ip.get('address')
    if ip_address:
        stateful_ip_to_patch.ipAddress = _CreateIpAddress(messages, ip_address)
    return stateful_ip_to_patch