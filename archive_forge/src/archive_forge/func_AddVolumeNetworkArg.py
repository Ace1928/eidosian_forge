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
def AddVolumeNetworkArg(parser, required=True):
    """Adds a --network arg to the given parser.

  Args:
    parser: argparse parser.
    required: bool whether arg is required or not
  """
    network_arg_spec = {'name': str, 'psa-range': str}
    network_help = "        Network configuration for a Cloud NetApp Files Volume. Specifying\n        `psa-range` is optional.\n        *name*::: The name of the Google Compute Engine\n        [VPC network](/compute/docs/networks-and-firewalls#networks) to which\n        the volume is connected.\n        *psa-range*::: The `psa-range` is the name of the allocated range of the\n        Private Service Access connection. The range you specify can't\n        overlap with either existing subnets or assigned IP address ranges for\n        other Cloud NetApp Files Volumes in the selected VPC network.\n  "
    parser.add_argument('--network', type=arg_parsers.ArgDict(spec=network_arg_spec, required_keys=['name']), required=required, help=network_help)