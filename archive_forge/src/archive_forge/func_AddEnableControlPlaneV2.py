from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddEnableControlPlaneV2(parser: parser_arguments.ArgumentInterceptor):
    """Adds a flag for enabling_control_plane_v2 field.

  Args:
    parser: The argparse parser to add the flag to.
  """
    control_plane_v2_mutex_group = parser.add_group(mutex=True)
    control_plane_v2_mutex_group.add_argument('--enable-control-plane-v2', help='If set, enable control plane v2.', action='store_true')
    control_plane_v2_mutex_group.add_argument('--disable-control-plane-v2', help='If set, disable control plane v2.', action='store_true')