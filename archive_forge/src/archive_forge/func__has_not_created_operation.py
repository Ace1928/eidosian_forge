from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
def _has_not_created_operation(result, retryer_state):
    """Takes TransferJob Apitools object and returns if it lacks an operation."""
    del retryer_state
    return not result.latestOperationName