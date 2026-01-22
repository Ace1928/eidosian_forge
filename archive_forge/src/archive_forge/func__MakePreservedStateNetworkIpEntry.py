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
def _MakePreservedStateNetworkIpEntry(messages, stateful_ip):
    """Prepares stateful ip preserved state entry."""
    auto_delete = (stateful_ip.get('auto-delete') or AutoDeleteFlag.NEVER).GetAutoDeleteEnumValue(messages.PreservedStatePreservedNetworkIp.AutoDeleteValueValuesEnum)
    address = None
    if stateful_ip.get('address'):
        ip_address = stateful_ip.get('address')
        address = _CreateIpAddress(messages, ip_address)
    return messages.PreservedStatePreservedNetworkIp(autoDelete=auto_delete, ipAddress=address)