from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddEdgeAvailabilityDomain(parser):
    """Adds edge-availability-domain flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--edge-availability-domain', choices=_EDGE_AVAILABILITY_DOMAIN_CHOICES, required=True, metavar='AVAILABILITY_DOMAIN', help='      Desired edge availability domain for this attachment:\n      `availability-domain-1`, `availability-domain-2`, `any`.\n\n      In each metro where the Partner can connect to Google, there are two sets\n      of redundant hardware. These sets are described as edge availability\n      domain 1 and 2. Within a metro, Google will only schedule maintenance in\n      one availability domain at a time. This guarantee does not apply to\n      availability domains outside the metro; Google may perform maintenance in\n      (say) New York availability domain 1 at the same time as Chicago\n      availability domain 1.\n      ')