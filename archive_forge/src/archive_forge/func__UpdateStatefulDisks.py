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
def _UpdateStatefulDisks(messages, per_instance_config, disks_to_update_dict, disks_to_remove_set, disk_getter):
    """Patch and return the updated list of stateful disks."""
    new_stateful_disks = []
    existing_disks = per_instance_config.preservedState.disks.additionalProperties if per_instance_config.preservedState.disks else []
    removed_stateful_disks_set = set()
    for current_stateful_disk in existing_disks:
        disk_name = current_stateful_disk.key
        if disk_name in disks_to_remove_set:
            removed_stateful_disks_set.add(disk_name)
            continue
        if disk_name in disks_to_update_dict:
            UpdateGA._PatchDiskData(messages, current_stateful_disk.value, disks_to_update_dict[disk_name])
            del disks_to_update_dict[disk_name]
        new_stateful_disks.append(current_stateful_disk)
    unremoved_stateful_disks_set = disks_to_remove_set.difference(removed_stateful_disks_set)
    if unremoved_stateful_disks_set:
        raise exceptions.InvalidArgumentException(parameter_name='--remove-stateful-disk', message='The following are invalid stateful disks: `{0}`'.format(','.join(unremoved_stateful_disks_set)))
    for update_stateful_disk in disks_to_update_dict.values():
        new_stateful_disks.append(instance_configs_messages.MakePreservedStateDiskEntry(messages=messages, stateful_disk_data=update_stateful_disk, disk_getter=disk_getter))
    return new_stateful_disks