from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkItemStatus(_messages.Message):
    """Conveys a worker's progress through the work described by a WorkItem.

  Fields:
    completed: True if the WorkItem was completed (successfully or
      unsuccessfully).
    counterUpdates: Worker output counters for this WorkItem.
    dynamicSourceSplit: See documentation of stop_position.
    errors: Specifies errors which occurred during processing. If errors are
      provided, and completed = true, then the WorkItem is considered to have
      failed.
    metricUpdates: DEPRECATED in favor of counter_updates.
    progress: DEPRECATED in favor of reported_progress.
    reportIndex: The report index. When a WorkItem is leased, the lease will
      contain an initial report index. When a WorkItem's status is reported to
      the system, the report should be sent with that report index, and the
      response will contain the index the worker should use for the next
      report. Reports received with unexpected index values will be rejected
      by the service. In order to preserve idempotency, the worker should not
      alter the contents of a report, even if the worker must submit the same
      report multiple times before getting back a response. The worker should
      not submit a subsequent report until the response for the previous
      report had been received from the service.
    reportedProgress: The worker's progress through this WorkItem.
    requestedLeaseDuration: Amount of time the worker requests for its lease.
    sourceFork: DEPRECATED in favor of dynamic_source_split.
    sourceOperationResponse: If the work item represented a
      SourceOperationRequest, and the work is completed, contains the result
      of the operation.
    stopPosition: A worker may split an active map task in two parts,
      "primary" and "residual", continuing to process the primary part and
      returning the residual part into the pool of available work. This event
      is called a "dynamic split" and is critical to the dynamic work
      rebalancing feature. The two obtained sub-tasks are called "parts" of
      the split. The parts, if concatenated, must represent the same input as
      would be read by the current task if the split did not happen. The exact
      way in which the original task is decomposed into the two parts is
      specified either as a position demarcating them (stop_position), or
      explicitly as two DerivedSources, if this task consumes a user-defined
      source type (dynamic_source_split). The "current" task is adjusted as a
      result of the split: after a task with range [A, B) sends a
      stop_position update at C, its range is considered to be [A, C), e.g.: *
      Progress should be interpreted relative to the new range, e.g. "75%
      completed" means "75% of [A, C) completed" * The worker should interpret
      proposed_stop_position relative to the new range, e.g. "split at 68%"
      should be interpreted as "split at 68% of [A, C)". * If the worker
      chooses to split again using stop_position, only stop_positions in [A,
      C) will be accepted. * Etc. dynamic_source_split has similar semantics:
      e.g., if a task with source S splits using dynamic_source_split into {P,
      R} (where P and R must be together equivalent to S), then subsequent
      progress and proposed_stop_position should be interpreted relative to P,
      and in a potential subsequent dynamic_source_split into {P', R'}, P' and
      R' must be together equivalent to P, etc.
    totalThrottlerWaitTimeSeconds: Total time the worker spent being throttled
      by external systems.
    workItemId: Identifies the WorkItem.
  """
    completed = _messages.BooleanField(1)
    counterUpdates = _messages.MessageField('CounterUpdate', 2, repeated=True)
    dynamicSourceSplit = _messages.MessageField('DynamicSourceSplit', 3)
    errors = _messages.MessageField('Status', 4, repeated=True)
    metricUpdates = _messages.MessageField('MetricUpdate', 5, repeated=True)
    progress = _messages.MessageField('ApproximateProgress', 6)
    reportIndex = _messages.IntegerField(7)
    reportedProgress = _messages.MessageField('ApproximateReportedProgress', 8)
    requestedLeaseDuration = _messages.StringField(9)
    sourceFork = _messages.MessageField('SourceFork', 10)
    sourceOperationResponse = _messages.MessageField('SourceOperationResponse', 11)
    stopPosition = _messages.MessageField('Position', 12)
    totalThrottlerWaitTimeSeconds = _messages.FloatField(13)
    workItemId = _messages.StringField(14)