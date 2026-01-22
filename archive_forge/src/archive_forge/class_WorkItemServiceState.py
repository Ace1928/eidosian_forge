from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkItemServiceState(_messages.Message):
    """The Dataflow service's idea of the current state of a WorkItem being
  processed by a worker.

  Messages:
    HarnessDataValue: Other data returned by the service, specific to the
      particular worker harness.

  Fields:
    completeWorkStatus: If set, a request to complete the work item with the
      given status. This will not be set to OK, unless supported by the
      specific kind of WorkItem. It can be used for the backend to indicate a
      WorkItem must terminate, e.g., for aborting work.
    harnessData: Other data returned by the service, specific to the
      particular worker harness.
    hotKeyDetection: A hot key is a symptom of poor data distribution in which
      there are enough elements mapped to a single key to impact pipeline
      performance. When present, this field includes metadata associated with
      any hot key.
    leaseExpireTime: Time at which the current lease will expire.
    metricShortId: The short ids that workers should use in subsequent metric
      updates. Workers should strive to use short ids whenever possible, but
      it is ok to request the short_id again if a worker lost track of it
      (e.g. if the worker is recovering from a crash). NOTE: it is possible
      that the response may have short ids for a subset of the metrics.
    nextReportIndex: The index value to use for the next report sent by the
      worker. Note: If the report call fails for whatever reason, the worker
      should reuse this index for subsequent report attempts.
    reportStatusInterval: New recommended reporting interval.
    splitRequest: The progress point in the WorkItem where the Dataflow
      service suggests that the worker truncate the task.
    suggestedStopPoint: DEPRECATED in favor of split_request.
    suggestedStopPosition: Obsolete, always empty.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class HarnessDataValue(_messages.Message):
        """Other data returned by the service, specific to the particular worker
    harness.

    Messages:
      AdditionalProperty: An additional property for a HarnessDataValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a HarnessDataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    completeWorkStatus = _messages.MessageField('Status', 1)
    harnessData = _messages.MessageField('HarnessDataValue', 2)
    hotKeyDetection = _messages.MessageField('HotKeyDetection', 3)
    leaseExpireTime = _messages.StringField(4)
    metricShortId = _messages.MessageField('MetricShortId', 5, repeated=True)
    nextReportIndex = _messages.IntegerField(6)
    reportStatusInterval = _messages.StringField(7)
    splitRequest = _messages.MessageField('ApproximateSplitRequest', 8)
    suggestedStopPoint = _messages.MessageField('ApproximateProgress', 9)
    suggestedStopPosition = _messages.MessageField('Position', 10)