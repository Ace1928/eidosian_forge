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
def _AddLoadBalancerPortConfig(bare_metal_load_balancer_config_group, is_update=False):
    """Adds flags to set port for load balancer.

  Args:
    bare_metal_load_balancer_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    if is_update:
        return None
    control_plane_load_balancer_port_config_group = bare_metal_load_balancer_config_group.add_group(help='Control plane load balancer port configuration.', required=True)
    control_plane_load_balancer_port_config_group.add_argument('--control-plane-load-balancer-port', required=True, help='Control plane load balancer port configuration.', type=int)