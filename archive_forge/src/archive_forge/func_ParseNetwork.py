from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseNetwork(network):
    """Parses a network name using configuration properties for fallback.

  Args:
    network: str, the network's ID, fully-qualified URL, or relative name

  Returns:
    googlecloudsdk.core.resources.Resource: a resource reference for the network
  """
    return resources.REGISTRY.Parse(network, params={'project': GetProject}, collection='compute.networks')