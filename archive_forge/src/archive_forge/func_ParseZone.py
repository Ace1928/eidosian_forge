from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseZone(zone):
    """Parses a zone name using configuration properties for fallback.

  Args:
    zone: str, the zone's ID, fully-qualified URL, or relative name

  Returns:
    googlecloudsdk.core.resources.Resource: a resource reference for the zone
  """
    return resources.REGISTRY.Parse(zone, params={'project': GetProject}, collection='compute.zones')