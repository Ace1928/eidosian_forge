from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseInstanceNetworkArg(network):
    if re.search(NETWORK_REGEX, network):
        return network
    project = properties.VALUES.core.project.GetOrFail()
    network_ref = resources.REGISTRY.Create('compute.networks', project=project, network=network)
    return network_ref.RelativeName()