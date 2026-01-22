from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddNetworkModeConfig(parser: parser_arguments.ArgumentInterceptor, is_update=False):
    """Adds network mode config related flags.

  Args:
    parser: The argparse parser to add the flag to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    network_config_mutex_group = parser.add_group(mutex=True, required=required, help='Populate one of the network configs.')
    _AddIslandModeCIDRConfig(network_config_mutex_group, is_update)