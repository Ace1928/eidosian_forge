from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import stateful_policy_utils as policy_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _GetUpdatedStatefulPolicyForDisks(self, client, current_stateful_policy, update_disks=None, remove_device_names=None):
    patched_disks_map = {}
    if remove_device_names:
        managed_instance_groups_utils.RegisterCustomStatefulDisksPatchEncoders(client)
    else:
        if current_stateful_policy and current_stateful_policy.preservedState and current_stateful_policy.preservedState.disks:
            current_disks = current_stateful_policy.preservedState.disks.additionalProperties
        else:
            current_disks = []
        patched_disks_map = {disk_entry.key: disk_entry for disk_entry in current_disks}
    for update_disk in update_disks or []:
        device_name = update_disk.get('device-name')
        updated_preserved_state_disk = policy_utils.MakeStatefulPolicyPreservedStateDiskEntry(client.messages, update_disk)
        if device_name in patched_disks_map:
            policy_utils.PatchStatefulPolicyDisk(patched_disks_map[device_name], updated_preserved_state_disk)
        else:
            patched_disks_map[device_name] = updated_preserved_state_disk
    for device_name in remove_device_names or []:
        patched_disks_map[device_name] = policy_utils.MakeDiskDeviceNullEntryForDisablingInPatch(client, device_name)
    stateful_disks = sorted([stateful_disk for _, stateful_disk in six.iteritems(patched_disks_map)], key=lambda x: x.key)
    return stateful_disks