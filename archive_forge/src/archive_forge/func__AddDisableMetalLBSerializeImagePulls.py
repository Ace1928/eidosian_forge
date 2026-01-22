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
def _AddDisableMetalLBSerializeImagePulls(metal_lb_kubelet_config_group, is_update=False):
    """Adds a flag to specify the enablement of serialize image pulls.

  Args:
    metal_lb_kubelet_config_group: The parent group to add the flags to.
    is_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if is_update:
        serialize_image_pulls_mutex_group = metal_lb_kubelet_config_group.add_group(mutex=True)
        surface = serialize_image_pulls_mutex_group
    else:
        surface = metal_lb_kubelet_config_group
    surface.add_argument('--disable-metal-lb-load-balancer-serialize-image-pulls', action='store_true', help='If set, prevent the Kubelet from pulling multiple images at a time.')
    if is_update:
        surface.add_argument('--enable-metal-lb-load-balancer-serialize-image-pulls', action='store_true', help='If set, enable the Kubelet to pull multiple images at a time.')