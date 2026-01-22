from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def ValidateUpdateStatefulPolicyParams(args, current_stateful_policy):
    """Check stateful properties of update request."""
    current_device_names = set(managed_instance_groups_utils.GetDeviceNamesFromStatefulPolicy(current_stateful_policy))
    update_disk_names = []
    if args.stateful_disk:
        ValidateStatefulDisksDict(args.stateful_disk, '--stateful-disk')
        update_disk_names = [stateful_disk.get('device-name') for stateful_disk in args.stateful_disk]
    if args.remove_stateful_disks:
        if any((args.remove_stateful_disks.count(x) > 1 for x in args.remove_stateful_disks)):
            raise exceptions.InvalidArgumentException(parameter_name='update', message='When removing device names from Stateful Policy, please provide each name exactly once.')
    update_set = set(update_disk_names)
    remove_set = set(args.remove_stateful_disks or [])
    intersection = update_set.intersection(remove_set)
    if intersection:
        raise exceptions.InvalidArgumentException(parameter_name='update', message='You cannot simultaneously add and remove the same device names {} to Stateful Policy.'.format(six.text_type(intersection)))
    not_current_device_names = remove_set - current_device_names
    if not_current_device_names:
        raise exceptions.InvalidArgumentException(parameter_name='update', message='Disks [{}] are not currently set as stateful, so they cannot be removed from Stateful Policy.'.format(six.text_type(not_current_device_names)))