from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSecurityGroupIds(parser, kind='control plane'):
    """Adds the --security-group-ids flag."""
    parser.add_argument('--security-group-ids', type=arg_parsers.ArgList(), metavar='SECURITY_GROUP_ID', help="IDs of additional security groups to add to the {}'s nodes.".format(kind))