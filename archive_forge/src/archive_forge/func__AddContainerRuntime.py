from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddContainerRuntime(bare_metal_workload_node_config_group):
    """Adds flags to set runtime for containers.

  Args:
    bare_metal_workload_node_config_group: The parent group to add the flags to.
  """
    bare_metal_workload_node_config_group.add_argument('--container-runtime', help='Container runtime which will be used in the bare metal user cluster.')