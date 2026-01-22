from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddPortMappingInfo(parser):
    """Adds port mapping info arguments for network endpoint groups."""
    help_text = '\n  Determines the spec of client port maping mode of this group.\n  Port Mapping is a use case in which NEG specifies routing by mapping client ports to destinations (e.g. ip and port).\n\n  *port-mapping-disabled*:::\n  Group should not be used for mapping client port to destination.\n\n  *client-port-per-endpoint*:::\n  For each endpoint there is exactly one client port.\n  '
    parser.add_argument('--client-port-mapping-mode', help=help_text)