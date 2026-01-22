from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectoryDnsArg(parser, required=True):
    """Adds a --dns arg to the given parser."""
    parser.add_argument('--dns', type=str, required=required, help='A comma separated list of DNS server IP addresses for the Active Directory domain.')