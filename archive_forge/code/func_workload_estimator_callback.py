from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.command_lib.storage import thread_messages
def workload_estimator_callback(status_queue, item_count, size=None):
    """Tracks expected item count and bytes for large operations.

  Information is sent to the status_queue, which will aggregate it
  for printing to the user. Useful for heavy operations like copy. For example,
  this sets the "100" in "copied 5/100 files."
  Arguments similar to thread_messages.WorkloadEstimatorMessage.

  Args:
    status_queue (multiprocessing.Queue): Reference to global queue.
    item_count (int): Number of items to add to workload estimation.
    size (int|None): Number of bytes to add to workload estimation.
  """
    status_queue.put(thread_messages.WorkloadEstimatorMessage(item_count=item_count, size=size))