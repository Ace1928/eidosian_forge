from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetPodAddressCidrBlocks(args):
    """Gets the value of --pod-address-cidr-blocks flag."""
    cidr_blocks = getattr(args, 'pod_address_cidr_blocks', None)
    return [cidr_blocks] if cidr_blocks else []