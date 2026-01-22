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
def HasSubsettingSubsetSizeArgs(args):
    """Returns true if request requires a Subsetting.subset_size field.

  Args:
    args: The arguments passed to the gcloud command.

  Returns:
    True if request requires a Subsetting.subset_size field.
  """
    return args.IsSpecified('subsetting_subset_size')