from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMaintenanceIntervalArgToParser(parser):
    """Add flag for adding maintenance interval to node group."""
    parser.add_argument('--maintenance-interval', choices=_MAINTENANCE_INTERVAL_CHOICES, type=lambda policy: policy.lower(), help='Specifies the frequency of planned maintenance events.')