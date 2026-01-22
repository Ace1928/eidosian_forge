from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_getter
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_messages
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
@staticmethod
def _PatchDiskData(messages, preserved_disk, update_disk_data):
    """Patch preserved disk according to arguments of `update_disk_data`."""
    auto_delete = update_disk_data.get('auto-delete')
    if update_disk_data.get('source'):
        preserved_disk.source = update_disk_data.get('source')
    if update_disk_data.get('mode'):
        preserved_disk.mode = instance_configs_messages.GetMode(messages=messages, mode=update_disk_data.get('mode'))
    if auto_delete:
        preserved_disk.autoDelete = auto_delete.GetAutoDeleteEnumValue(messages.PreservedStatePreservedDisk.AutoDeleteValueValuesEnum)
    return preserved_disk