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
def _AddNoProxyConfig(bare_metal_proxy_config_group):
    """Adds a flag to specify the address of the proxy server.

  Args:
    bare_metal_proxy_config_group: The parent group to add the flag to.
  """
    bare_metal_proxy_config_group.add_argument('--no-proxy', metavar='NO_PROXY', type=arg_parsers.ArgList(), help='List of IPs, hostnames, and domains that should skip the proxy.')