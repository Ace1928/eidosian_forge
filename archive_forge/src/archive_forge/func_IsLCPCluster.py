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
def IsLCPCluster(args):
    """Identify if the command is creating LCP cluster.

  Args:
    args: command line arguments.

  Returns:
    Boolean, indication of LCP cluster.
  """
    if flags.FlagIsExplicitlySet(args, 'control_plane_node_location') and flags.FlagIsExplicitlySet(args, 'control_plane_node_count') and (flags.FlagIsExplicitlySet(args, 'external_lb_ipv4_address_pools') or flags.FlagIsExplicitlySet(args, 'external_lb_address_pools')):
        return True
    return False