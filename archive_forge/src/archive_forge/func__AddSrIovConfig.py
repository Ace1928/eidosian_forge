from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddSrIovConfig(bare_metal_network_config_group, is_update=False):
    """Adds a flag to specify the SR-IOV Config.

  Args:
    bare_metal_network_config_group: The parent group to add the flags to.
    is_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    required = not is_update
    sr_iov_config_group = bare_metal_network_config_group.add_group(help='Anthos on bare metal standalone cluster SR-IOV configuration.', required=required)
    _AddEnableSrIovConfig(sr_iov_config_group, is_update)