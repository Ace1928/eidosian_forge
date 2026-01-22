from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerMessage(_messages.Message):
    """WorkerMessage provides information to the backend about a worker.

  Messages:
    LabelsValue: Labels are used to group WorkerMessages. For example, a
      worker_message about a particular container might have the labels: {
      "JOB_ID": "2015-04-22", "WORKER_ID": "wordcount-vm-2015..."
      "CONTAINER_TYPE": "worker", "CONTAINER_ID": "ac1234def"} Label tags
      typically correspond to Label enum values. However, for ease of
      development other strings can be used as tags. LABEL_UNSPECIFIED should
      not be used here.

  Fields:
    dataSamplingReport: Optional. Contains metrics related to go/dataflow-
      data-sampling-telemetry.
    labels: Labels are used to group WorkerMessages. For example, a
      worker_message about a particular container might have the labels: {
      "JOB_ID": "2015-04-22", "WORKER_ID": "wordcount-vm-2015..."
      "CONTAINER_TYPE": "worker", "CONTAINER_ID": "ac1234def"} Label tags
      typically correspond to Label enum values. However, for ease of
      development other strings can be used as tags. LABEL_UNSPECIFIED should
      not be used here.
    perWorkerMetrics: System defined metrics for this worker.
    streamingScalingReport: Contains per-user worker telemetry used in
      streaming autoscaling.
    time: The timestamp of the worker_message.
    workerHealthReport: The health of a worker.
    workerLifecycleEvent: Record of worker lifecycle events.
    workerMessageCode: A worker message code.
    workerMetrics: Resource metrics reported by workers.
    workerShutdownNotice: Shutdown notice by workers.
    workerThreadScalingReport: Thread scaling information reported by workers.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels are used to group WorkerMessages. For example, a worker_message
    about a particular container might have the labels: { "JOB_ID":
    "2015-04-22", "WORKER_ID": "wordcount-vm-2015..." "CONTAINER_TYPE":
    "worker", "CONTAINER_ID": "ac1234def"} Label tags typically correspond to
    Label enum values. However, for ease of development other strings can be
    used as tags. LABEL_UNSPECIFIED should not be used here.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dataSamplingReport = _messages.MessageField('DataSamplingReport', 1)
    labels = _messages.MessageField('LabelsValue', 2)
    perWorkerMetrics = _messages.MessageField('PerWorkerMetrics', 3)
    streamingScalingReport = _messages.MessageField('StreamingScalingReport', 4)
    time = _messages.StringField(5)
    workerHealthReport = _messages.MessageField('WorkerHealthReport', 6)
    workerLifecycleEvent = _messages.MessageField('WorkerLifecycleEvent', 7)
    workerMessageCode = _messages.MessageField('WorkerMessageCode', 8)
    workerMetrics = _messages.MessageField('ResourceUtilizationReport', 9)
    workerShutdownNotice = _messages.MessageField('WorkerShutdownNotice', 10)
    workerThreadScalingReport = _messages.MessageField('WorkerThreadScalingReport', 11)