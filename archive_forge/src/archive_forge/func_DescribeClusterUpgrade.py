from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container.fleet.scopes.rollout_sequencing import base
from googlecloudsdk.core import log
def DescribeClusterUpgrade(response, args):
    """Adds Cluster Upgrade Feature information to describe Scope request.

  This is a modify_request_hook for gcloud declarative YAML.

  Args:
    response: Scope message.
    args: command line arguments.

  Returns:
    response with optional Cluster Upgrade Feature information
  """
    cmd = base.DescribeCommand(args)
    if cmd.IsClusterUpgradeRequest():
        feature = cmd.GetFeature()
        return cmd.GetScopeWithClusterUpgradeInfo(response, feature)
    return response