from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
def _PossiblyCompleteStage(self, condition, message):
    """Complete the stage if it's not already complete.

    Make sure the necessary internal bookkeeping is done.

    Args:
      condition: str, The name of the condition whose stage should be completed.
      message: str, The detailed message for the condition.

    Returns:
      bool: True if stage was completed, False if no action taken
    """
    if condition not in self._tracker or self._tracker.IsComplete(condition):
        return False
    if not self._tracker.IsRunning(condition):
        return False
    self._RecordConditionComplete(condition)
    self._StartUnblocked()
    self._tracker.CompleteStage(condition, message)
    return True