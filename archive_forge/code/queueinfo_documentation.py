from __future__ import absolute_import
from __future__ import unicode_literals
import os
Populates a `TaskQueueRetryParameters` from a `queueinfo.RetryParameters`.

  Args:
    retry: A `queueinfo.RetryParameters` that is read from `queue.yaml` that
        describes the queue's retry parameters.

  Returns:
    A `taskqueue_service_pb.TaskQueueRetryParameters` proto populated with the
    data from `retry`.

  Raises:
    MalformedQueueConfiguration: If the retry parameters are invalid.
  