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
def HasIpAddressSelectionPolicyArgs(args):
    """Returns true if request requires an IP address selection policy.

  Args:
    args: The arguments passed to the gcloud command.

  Returns:
    True if request requires an IP address selection policy.
  """
    return args.IsSpecified('ip_address_selection_policy')