from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectorySecurityOperatorsArg(parser):
    """Adds a --security-operators arg to the given parser."""
    parser.add_argument('--security-operators', type=arg_parsers.ArgList(element_type=str), metavar='SECURITY_OPERATOR', help='Domain users to be given the Security privilege.')