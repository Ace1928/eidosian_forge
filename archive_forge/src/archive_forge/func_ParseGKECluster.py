from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def ParseGKECluster(gke_cluster):
    """Parse GKE cluster's location and name.

  Args:
   gke_cluster: GKE cluster sepecified in format {location}/{cluster_name}

  Returns:
    cluster's location, and name
  """
    rgx = '(.*)\\/(.*)'
    cluster_matcher = re.search(rgx, gke_cluster)
    if cluster_matcher is not None:
        return (cluster_matcher.group(1), cluster_matcher.group(2))
    raise exceptions.Error('argument --gke-cluster: {} is invalid. --gke-cluster must be of format: `{{REGION OR ZONE}}/{{CLUSTER_NAME`}}`'.format(gke_cluster))