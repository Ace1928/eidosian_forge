from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeProtocolsArg(parser, required=True):
    """Adds the Protocols arg to the arg parser."""
    parser.add_argument('--protocols', type=arg_parsers.ArgList(min_length=1, element_type=str), required=required, metavar='PROTOCOL', help='Type of File System protocols for the Cloud NetApp Files Volume. Valid component values are:\n            `NFSV3`, `NFSV4`, `SMB`.')