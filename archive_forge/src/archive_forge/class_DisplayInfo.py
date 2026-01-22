from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.core.util import times
class DisplayInfo(object):
    """Information about a job displayed in command output.

  Fields:
    id: the job ID
    name: the job name
    type: one of 'batch', 'streaming'
    state: string representing the current job status
    creationTime: in the form yyyy-mm-dd hh:mm:ss
    stateTime: in the form yyyy-mm-dd hh:mm:ss
    location: the job's regional endpoint
  """

    def __init__(self, job):
        self.id = job.id
        self.name = job.name
        self.type = DisplayInfo._JobTypeForJob(job.type)
        self.state = DisplayInfo._StatusForJob(job.currentState)
        self.location = job.location
        self.stateTime = FormatDateTime(job.currentStateTime)
        self.creationTime = FormatDateTime(job.createTime)

    @staticmethod
    def _JobTypeForJob(job_type):
        """Return a string describing the job type.

    Args:
      job_type: The job type enum
    Returns:
      string describing the job type
    """
        type_value_enum = apis.GetMessagesModule().Job.TypeValueValuesEnum
        value_map = {type_value_enum.JOB_TYPE_BATCH: 'Batch', type_value_enum.JOB_TYPE_STREAMING: 'Streaming'}
        return value_map.get(job_type, 'Unknown')

    @staticmethod
    def _StatusForJob(job_state):
        """Return a string describing the job state.

    Args:
      job_state: The job state enum
    Returns:
      string describing the job state
    """
        state_value_enum = apis.GetMessagesModule().Job.CurrentStateValueValuesEnum
        value_map = {state_value_enum.JOB_STATE_CANCELLED: 'Cancelled', state_value_enum.JOB_STATE_CANCELLING: 'Cancelling', state_value_enum.JOB_STATE_DONE: 'Done', state_value_enum.JOB_STATE_DRAINED: 'Drained', state_value_enum.JOB_STATE_DRAINING: 'Draining', state_value_enum.JOB_STATE_FAILED: 'Failed', state_value_enum.JOB_STATE_PENDING: 'Pending', state_value_enum.JOB_STATE_QUEUED: 'Queued', state_value_enum.JOB_STATE_RUNNING: 'Running', state_value_enum.JOB_STATE_STOPPED: 'Stopped', state_value_enum.JOB_STATE_UPDATED: 'Updated'}
        return value_map.get(job_state, 'Unknown')