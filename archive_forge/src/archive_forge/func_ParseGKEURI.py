from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def ParseGKEURI(gke_uri):
    """The GKE resource URI can be of following types: zonal, regional or generic.

  zonal - */projects/{project_id}/zones/{zone}/clusters/{cluster_name}
  regional - */projects/{project_id}/regions/{zone}/clusters/{cluster_name}
  generic - */projects/{project_id}/locations/{zone}/clusters/{cluster_name}

  The expected patterns are matched to extract the cluster location and name.
  Args:
   gke_uri: GKE resource URI, e.g., https://container.googleapis.com/v1/
     projects/my-project/zones/us-west2-c/clusters/test1

  Returns:
    cluster's project, location, and name
  """
    zonal_uri_pattern = '.*\\/projects\\/(.*)\\/zones\\/(.*)\\/clusters\\/(.*)'
    regional_uri_pattern = '.*\\/projects\\/(.*)\\/regions\\/(.*)\\/clusters\\/(.*)'
    location_uri_pattern = '.*\\/projects\\/(.*)\\/locations\\/(.*)\\/clusters\\/(.*)'
    zone_matcher = re.search(zonal_uri_pattern, gke_uri)
    if zone_matcher is not None:
        return (zone_matcher.group(1), zone_matcher.group(2), zone_matcher.group(3))
    region_matcher = re.search(regional_uri_pattern, gke_uri)
    if region_matcher is not None:
        return (region_matcher.group(1), region_matcher.group(2), region_matcher.group(3))
    location_matcher = re.search(location_uri_pattern, gke_uri)
    if location_matcher is not None:
        return (location_matcher.group(1), location_matcher.group(2), location_matcher.group(3))
    raise exceptions.Error('argument --gke-uri: {} is invalid. --gke-uri must be of format: `https://container.googleapis.com/v1/projects/my-project/locations/us-central1-a/clusters/my-cluster`. You can use command: `gcloud container clusters list --uri` to view the current GKE clusters in your project.'.format(gke_uri))