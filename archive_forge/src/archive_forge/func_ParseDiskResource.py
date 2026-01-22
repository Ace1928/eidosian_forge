from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def ParseDiskResource(resources, name, project, zone, type_):
    """Parses disk resources.

  Project and zone are ignored if a fully-qualified resource name is given, i.e.
    - https://compute.googleapis.com/compute/v1/projects/my-project
          /zones/us-central1-a/disks/disk-1
    - projects/my-project/zones/us-central1-a/disks/disk-1

  If project and zone cannot be parsed, we will use the given project and zone
  as fallbacks.

  Args:
    resources: resources.Registry, The resource registry
    name: str, name of the disk.
    project: str, project of the disk.
    zone: str, zone of the disk.
    type_: ScopeEnum, type of the disk.

  Returns:
    A disk resource.
  """
    if type_ == compute_scopes.ScopeEnum.REGION:
        return resources.Parse(name, collection='compute.regionDisks', params={'project': project, 'region': utils.ZoneNameToRegionName(zone)})
    else:
        return resources.Parse(name, collection='compute.disks', params={'project': project, 'zone': zone})