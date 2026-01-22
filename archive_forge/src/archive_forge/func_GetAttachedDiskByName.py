from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.console import console_io
def GetAttachedDiskByName(self, resources, name, instance_ref, instance):
    """Gets an attached disk with the specified disk name.

    First, we attempt to parse the provided disk name to find the possible disks
    that it may describe. Next, we iterate over the attached disks to find the
    ones that match the possible disks.

    If the disk can match multiple disks, we prompt the user to choose one.

    Args:
      resources: resources.Registry, The resource registry
      name: str, name of the attached disk.
      instance_ref: Reference of the instance instance.
      instance: Instance object.

    Returns:
      An attached disk object.

    Raises:
      exceptions.ArgumentError: If a disk with name cannot be found attached to
          the instance or if the user does not choose a specific disk when
          prompted.
    """
    possible_disks = self._GetPossibleDisks(resources, name, instance_ref)
    matched_attached_disks = []
    for attached_disk in instance.disks:
        parsed_disk = instance_utils.ParseDiskResourceFromAttachedDisk(resources, attached_disk)
        for d in possible_disks:
            if d and d.RelativeName() == parsed_disk.RelativeName():
                matched_attached_disks.append(attached_disk)
    if not matched_attached_disks:
        raise compute_exceptions.ArgumentError('Disk [{}] is not attached to instance [{}] in zone [{}].'.format(name, instance_ref.instance, instance_ref.zone))
    elif len(matched_attached_disks) == 1:
        return matched_attached_disks[0]
    choice_names = []
    for attached_disk in matched_attached_disks:
        disk_ref = instance_utils.ParseDiskResourceFromAttachedDisk(resources, attached_disk)
        choice_names.append(disk_ref.RelativeName())
    idx = console_io.PromptChoice(options=choice_names, message='[{}] matched multiple disks. Choose one:'.format(name))
    if idx is None:
        raise compute_exceptions.ArgumentError('Found multiple disks matching [{}] attached to instance [{}] in zone [{}].'.format(name, instance_ref.instance, instance_ref.zone))
    return matched_attached_disks[idx]