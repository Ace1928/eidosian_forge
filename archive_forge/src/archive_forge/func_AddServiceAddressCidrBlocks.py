from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddServiceAddressCidrBlocks(parser):
    """Add the --service-address-cidr-blocks flag."""
    parser.add_argument('--service-address-cidr-blocks', required=True, help='IP address range for the services IPs in CIDR notation (e.g. 10.0.0.0/8).')