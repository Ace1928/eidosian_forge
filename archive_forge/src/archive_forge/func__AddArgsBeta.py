from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.deployment_resource_pools import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import resources
def _AddArgsBeta(parser):
    """Adding deployment resource pool arguments from CLI.

  Args:
    parser: argparse.ArgumentParser, cli argument parser

  Returns:
    None
  """
    parser.display_info.AddFormat(_DEFAULT_FORMAT)
    parser.display_info.AddUriFunc(_GetUri)
    flags.AddRegionResourceArg(parser, 'to list deployment resource pools', prompt_func=region_util.PromptForDeploymentResourcePoolSupportedRegion)