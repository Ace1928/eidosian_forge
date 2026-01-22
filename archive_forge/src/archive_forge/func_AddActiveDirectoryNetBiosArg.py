from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectoryNetBiosArg(parser, required=True):
    """Adds a --net-bios-prefix arg to the given parser."""
    parser.add_argument('--net-bios-prefix', type=str, required=required, help='NetBIOS prefix name of the server.')