from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def ClusterShortName(resource_name):
    """Get the name part of a cluster membership's full resource name.

  For example, "projects/123/locations/global/memberships/cluster2" returns
  "cluster2".

  Args:
    resource_name: A cluster's full resource name.

  Raises:
    ValueError: If the full resource name was not well-formatted.

  Returns:
    The cluster's short name.
  """
    match = re.search(CLUSTER_NAME_SELECTOR, resource_name)
    if match:
        return match.group(1)
    raise ValueError('The cluster membership resource name must match "%s"' % (CLUSTER_NAME_SELECTOR,))