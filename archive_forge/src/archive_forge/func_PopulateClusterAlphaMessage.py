from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.container import admin_users
from googlecloudsdk.command_lib.edge_cloud.container import fleet
from googlecloudsdk.command_lib.edge_cloud.container import resource_args
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import resources
def PopulateClusterAlphaMessage(req, messages, args):
    """Filled the Alpha cluster message from command arguments.

  Args:
    req: create cluster request message.
    messages: message module of edgecontainer cluster.
    args: command line arguments.
  """
    if flags.FlagIsExplicitlySet(args, 'cluster_ipv6_cidr'):
        req.cluster.networking.clusterIpv6CidrBlocks = [args.cluster_ipv6_cidr]
    if flags.FlagIsExplicitlySet(args, 'services_ipv6_cidr'):
        req.cluster.networking.servicesIpv6CidrBlocks = [args.services_ipv6_cidr]
    if flags.FlagIsExplicitlySet(args, 'external_lb_ipv6_address_pools'):
        req.cluster.externalLoadBalancerIpv6AddressPools = args.external_lb_ipv6_address_pools
    resource_args.SetSystemAddonsConfig(args, req)
    if flags.FlagIsExplicitlySet(args, 'offline_reboot_ttl'):
        req.cluster.survivabilityConfig = messages.SurvivabilityConfig()
        req.cluster.survivabilityConfig.offlineRebootTtl = json.dumps(args.offline_reboot_ttl) + 's'
    resource_args.SetExternalLoadBalancerAddressPoolsConfig(args, req)