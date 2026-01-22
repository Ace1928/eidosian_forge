from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskRun(_messages.Message):
    """Message describing TaskRun object

  Enums:
    TaskRunStatusValueValuesEnum: Taskrun status the user can provide. Used
      for cancellation.

  Messages:
    AnnotationsValue: User annotations. See
      https://google.aip.dev/128#annotations
    GcbParamsValue: Output only. GCB default params.

  Fields:
    annotations: User annotations. See https://google.aip.dev/128#annotations
    completionTime: Output only. Time the task completed.
    conditions: Output only. Kubernetes Conditions convention for PipelineRun
      status and error.
    createTime: Output only. Time at which the request to create the `TaskRun`
      was received.
    etag: Needed for declarative-friendly resources.
    gcbParams: Output only. GCB default params.
    name: Output only. The 'TaskRun' name with format:
      `projects/{project}/locations/{location}/taskRuns/{task_run}`
    params: Params is a list of parameter names and values.
    pipelineRun: Output only. Name of the parent PipelineRun. If it is a
      standalone TaskRun (no parent), this field will not be set.
    provenance: Optional. Provenance configuration.
    record: Output only. The `Record` of this `TaskRun`. Format: `projects/{pr
      oject}/locations/{location}/results/{result_id}/records/{record_id}`
    resolvedTaskSpec: Output only. The exact TaskSpec used to instantiate the
      run.
    results: Output only. List of results written out by the task's containers
    security: Optional. Security configuration.
    serviceAccount: Required. Service account used in the task. Deprecated;
      please use security.service_account instead.
    sidecars: Output only. State of each Sidecar in the TaskSpec.
    startTime: Output only. Time the task is actually started.
    statusMessage: Optional. Output only. Status message for cancellation.
      +optional
    steps: Output only. Steps describes the state of each build step
      container.
    taskRef: TaskRef refer to a specific instance of a task.
    taskRunStatus: Taskrun status the user can provide. Used for cancellation.
    taskSpec: TaskSpec contains the Spec to instantiate a TaskRun.
    timeout: Time after which the task times out. Defaults to 1 hour. If you
      set the timeout to 0, the TaskRun will have no timeout and will run
      until it completes successfully or fails from an error.
    uid: Output only. A unique identifier for the `TaskRun`.
    updateTime: Output only. Time at which the request to update the `TaskRun`
      was received.
    worker: Optional. Worker configuration.
    workerPool: Output only. The WorkerPool used to run this TaskRun.
    workspaces: Workspaces is a list of WorkspaceBindings from volumes to
      workspaces.
  """

    class TaskRunStatusValueValuesEnum(_messages.Enum):
        """Taskrun status the user can provide. Used for cancellation.

    Values:
      TASK_RUN_STATUS_UNSPECIFIED: Default enum type; should not be used.
      TASK_RUN_CANCELLED: Cancelled status.
    """
        TASK_RUN_STATUS_UNSPECIFIED = 0
        TASK_RUN_CANCELLED = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """User annotations. See https://google.aip.dev/128#annotations

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class GcbParamsValue(_messages.Message):
        """Output only. GCB default params.

    Messages:
      AdditionalProperty: An additional property for a GcbParamsValue object.

    Fields:
      additionalProperties: Additional properties of type GcbParamsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a GcbParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    completionTime = _messages.StringField(2)
    conditions = _messages.MessageField('GoogleDevtoolsCloudbuildV2Condition', 3, repeated=True)
    createTime = _messages.StringField(4)
    etag = _messages.StringField(5)
    gcbParams = _messages.MessageField('GcbParamsValue', 6)
    name = _messages.StringField(7)
    params = _messages.MessageField('Param', 8, repeated=True)
    pipelineRun = _messages.StringField(9)
    provenance = _messages.MessageField('Provenance', 10)
    record = _messages.StringField(11)
    resolvedTaskSpec = _messages.MessageField('TaskSpec', 12)
    results = _messages.MessageField('TaskRunResult', 13, repeated=True)
    security = _messages.MessageField('Security', 14)
    serviceAccount = _messages.StringField(15)
    sidecars = _messages.MessageField('SidecarState', 16, repeated=True)
    startTime = _messages.StringField(17)
    statusMessage = _messages.StringField(18)
    steps = _messages.MessageField('StepState', 19, repeated=True)
    taskRef = _messages.MessageField('TaskRef', 20)
    taskRunStatus = _messages.EnumField('TaskRunStatusValueValuesEnum', 21)
    taskSpec = _messages.MessageField('TaskSpec', 22)
    timeout = _messages.StringField(23)
    uid = _messages.StringField(24)
    updateTime = _messages.StringField(25)
    worker = _messages.MessageField('Worker', 26)
    workerPool = _messages.StringField(27)
    workspaces = _messages.MessageField('WorkspaceBinding', 28, repeated=True)