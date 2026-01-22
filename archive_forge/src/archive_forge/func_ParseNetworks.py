from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def ParseNetworks(value, project, version):
    """Build a list of PolicyNetworks or ResponsePolicyNetworks from command line args."""
    if not value:
        return []
    registry = api_util.GetRegistry(version)
    networks = [registry.Parse(network_name, collection='compute.networks', params={'project': project}) for network_name in value]
    return networks