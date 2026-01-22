from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def HasFailoverPolicyArgs(args):
    """Returns true if at least one of the failover policy args is defined.

  Args:
    args: The arguments passed to the gcloud command.
  """
    if args.IsSpecified('connection_drain_on_failover') or args.IsSpecified('drop_traffic_if_unhealthy') or args.IsSpecified('failover_ratio'):
        return True
    else:
        return False