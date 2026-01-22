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
def _GetPossibleDisks(self, resources, name, instance_ref):
    """Gets the possible disks that match the provided disk name.

    First, we attempt to parse the provided disk name as a regional and as a
    zonal disk. Next, we iterate over the attached disks to find the ones that
    match the parsed regional and zonal disks.

    If the disk can match either a zonal or regional disk, we prompt the user to
    choose one.

    Args:
      resources: resources.Registry, The resource registry
      name: str, name of the disk.
      instance_ref: Reference of the instance instance.

    Returns:
      List of possible disks references that possibly match the provided disk
          name.
    """
    possible_disks = []
    try:
        regional_disk = instance_utils.ParseDiskResource(resources, name, instance_ref.project, instance_ref.zone, compute_scopes.ScopeEnum.REGION)
        possible_disks.append(regional_disk)
    except cloud_resources.WrongResourceCollectionException:
        pass
    try:
        zonal_disk = instance_utils.ParseDiskResource(resources, name, instance_ref.project, instance_ref.zone, compute_scopes.ScopeEnum.ZONE)
        possible_disks.append(zonal_disk)
    except cloud_resources.WrongResourceCollectionException:
        pass
    return possible_disks