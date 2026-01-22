from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddClearSshEc2KeyPair(parser, kind):
    """Adds the --clear-ssh-ec2-key-pair flag."""
    parser.add_argument('--clear-ssh-ec2-key-pair', action='store_true', default=None, help="Clear the EC2 key pair authorized to login to the {}'s nodes.".format(kind))