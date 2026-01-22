from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1Job(_messages.Message):
    """Definition of the job information maintained by the pipeline. Fields in
  this entity are retrieved from the executor API (e.g. Dataflow API).

  Enums:
    StateValueValuesEnum: The current state of the job.

  Fields:
    createTime: Output only. The time of job creation.
    dataflowJobDetails: All the details that are specific to a Dataflow job.
    endTime: Output only. The time of job termination. This is absent if the
      job is still running.
    id: Output only. The internal ID for the job.
    name: Required. The fully qualified resource name for the job.
    state: The current state of the job.
    status: Status capturing any error code or message related to job creation
      or execution.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the job.

    Values:
      STATE_UNSPECIFIED: The job state isn't specified.
      STATE_PENDING: The job is waiting to start execution.
      STATE_RUNNING: The job is executing.
      STATE_DONE: The job has finished execution successfully.
      STATE_FAILED: The job has finished execution with a failure.
      STATE_CANCELLED: The job has been terminated upon user request.
    """
        STATE_UNSPECIFIED = 0
        STATE_PENDING = 1
        STATE_RUNNING = 2
        STATE_DONE = 3
        STATE_FAILED = 4
        STATE_CANCELLED = 5
    createTime = _messages.StringField(1)
    dataflowJobDetails = _messages.MessageField('GoogleCloudDatapipelinesV1DataflowJobDetails', 2)
    endTime = _messages.StringField(3)
    id = _messages.StringField(4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    status = _messages.MessageField('GoogleRpcStatus', 7)