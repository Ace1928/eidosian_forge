from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskInstance(_messages.Message):
    """Task instance in a DAG run.

  Enums:
    StateValueValuesEnum: Task instance state.

  Fields:
    dagId: The DAG ID of the DAG whose execution is described by the DAG run
      the taskInstance belongs to.
    dagRunId: The DAG run ID the task instance belongs to.
    endDate: Timestamp when the task instance finished execution.
    executionDate: Execution date for the task.
    externalExecutorId: ID of the external executor.
    hostname: Hostname of the machine or pod the task runs on.
    id: The task instance ID. It is the same as the task ID of a DAG.
    isDynamicallyMapped: Whether this TaskInstance is dynamically mapped.
    mapIndex: If is_dynamically_mapped is set to true, this field contains
      index of the dynamically-mapped TaskInstance, If is_dynamically_mapped
      is set to false, this field has no meaning.
    maxTries: The number of tries that should be performed before failing the
      task.
    name: Required. The resource name of the task instance, in the form: "proj
      ects/{project_id}/locations/{location_id}/environments/{environment_id}/
      dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_instance_id_with_
      optional_map_index}".
    pool: The slot pool this task runs in.
    priorityWeight: Priority weight of this task against other tasks.
    queue: Which queue to target when running this task.
    queuedDttm: Timestamp when the task was queued.
    startDate: Timestamp when the task instance started execution.
    state: Task instance state.
    taskId: The task instance ID. It is the same as the task ID in the DAG.
    taskType: The operator used in this task.
    tryNumber: The try number that this task number will be when it is
      actually run.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Task instance state.

    Values:
      STATE_UNSPECIFIED: The state of the task instance is unknown.
      SUCCEEDED: Task execution succeeded.
      FAILED: Task execution failed.
      UPSTREAM_FAILED: Upstream task failed.
      SKIPPED: Task skipped.
      UP_FOR_RETRY: Task waiting to be retried.
      UP_FOR_RESCHEDULE: Task waiting to be rescheduled.
      QUEUED: Task queued.
      SCHEDULED: Task scheduled for execution.
      SENSING: Task in sensing mode.
      REMOVED: Task vanished from DAG before it ran.
      RUNNING: Task is executing.
      SHUTDOWN: External request to shut down (e.g. marked failed when
        running).
      RESTARTING: External request to restart (e.g. cleared when running).
      DEFERRED: Deferrable operator waiting on a trigger.
    """
        STATE_UNSPECIFIED = 0
        SUCCEEDED = 1
        FAILED = 2
        UPSTREAM_FAILED = 3
        SKIPPED = 4
        UP_FOR_RETRY = 5
        UP_FOR_RESCHEDULE = 6
        QUEUED = 7
        SCHEDULED = 8
        SENSING = 9
        REMOVED = 10
        RUNNING = 11
        SHUTDOWN = 12
        RESTARTING = 13
        DEFERRED = 14
    dagId = _messages.StringField(1)
    dagRunId = _messages.StringField(2)
    endDate = _messages.StringField(3)
    executionDate = _messages.StringField(4)
    externalExecutorId = _messages.StringField(5)
    hostname = _messages.StringField(6)
    id = _messages.StringField(7)
    isDynamicallyMapped = _messages.BooleanField(8)
    mapIndex = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    maxTries = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    name = _messages.StringField(11)
    pool = _messages.StringField(12)
    priorityWeight = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    queue = _messages.StringField(14)
    queuedDttm = _messages.StringField(15)
    startDate = _messages.StringField(16)
    state = _messages.EnumField('StateValueValuesEnum', 17)
    taskId = _messages.StringField(18)
    taskType = _messages.StringField(19)
    tryNumber = _messages.IntegerField(20, variant=_messages.Variant.INT32)