from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetInterconnectAttachmentsFlag():
    """Returns the flag for interconnect attachments (VLAN attachments) associated with a VPN gateway."""
    return base.Argument('--interconnect-attachments', type=arg_parsers.ArgList(max_length=2), required=False, metavar='INTERCONNECT_ATTACHMENTS', help='      Names of interconnect attachments (VLAN attachments) associated with the\n      VPN gateway interfaces. You must specify this field when using a VPN gateway\n      for HA VPN over Cloud Interconnect. Otherwise, this field is optional.\n\n      For example,\n      `--interconnect-attachments attachment-a-zone1,attachment-a-zone2`\n      associates VPN gateway with attachment from zone1 on interface 0 and with\n      attachment from zone2 on interface 1.\n      ')