from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineRun(_messages.Message):
    """Message describing PipelineRun object

  Enums:
    PipelineRunStatusValueValuesEnum: Pipelinerun status the user can provide.
      Used for cancellation.

  Messages:
    AnnotationsValue: User annotations. See
      https://google.aip.dev/128#annotations
    GcbParamsValue: Output only. GCB default params.

  Fields:
    annotations: User annotations. See https://google.aip.dev/128#annotations
    childReferences: Output only. List of TaskRun and Run names and
      PipelineTask names for children of this PipelineRun.
    completionTime: Output only. Time the pipeline completed.
    conditions: Output only. Kubernetes Conditions convention for PipelineRun
      status and error.
    createTime: Output only. Time at which the request to create the
      `PipelineRun` was received.
    etag: Needed for declarative-friendly resources.
    finallyStartTime: Output only. FinallyStartTime is when all non-finally
      tasks have been completed and only finally tasks are being executed.
      +optional
    gcbParams: Output only. GCB default params.
    name: Output only. The `PipelineRun` name with format
      `projects/{project}/locations/{location}/pipelineRuns/{pipeline_run}`
    params: Params is a list of parameter names and values.
    pipelineRef: PipelineRef refer to a specific instance of a Pipeline.
    pipelineRunStatus: Pipelinerun status the user can provide. Used for
      cancellation.
    pipelineSpec: PipelineSpec defines the desired state of Pipeline.
    provenance: Optional. Provenance configuration.
    record: Output only. The `Record` of this `PipelineRun`. Format: `projects
      /{project}/locations/{location}/results/{result_id}/records/{record_id}`
    resolvedPipelineSpec: Output only. The exact PipelineSpec used to
      instantiate the run.
    results: Optional. Output only. List of results written out by the
      pipeline's containers
    security: Optional. Security configuration.
    serviceAccount: Service account used in the Pipeline. Deprecated; please
      use security.service_account instead.
    skippedTasks: Output only. List of tasks that were skipped due to when
      expressions evaluating to false.
    startTime: Output only. Time the pipeline is actually started.
    timeouts: Time after which the Pipeline times out. Currently three keys
      are accepted in the map pipeline, tasks and finally with
      Timeouts.pipeline >= Timeouts.tasks + Timeouts.finally
    uid: Output only. A unique identifier for the `PipelineRun`.
    updateTime: Output only. Time at which the request to update the
      `PipelineRun` was received.
    worker: Optional. Worker configuration.
    workerPool: Output only. The WorkerPool used to run this PipelineRun.
    workflow: Output only. The Workflow used to create this PipelineRun.
    workspaces: Workspaces is a list of WorkspaceBindings from volumes to
      workspaces.
  """

    class PipelineRunStatusValueValuesEnum(_messages.Enum):
        """Pipelinerun status the user can provide. Used for cancellation.

    Values:
      PIPELINE_RUN_STATUS_UNSPECIFIED: Default enum type; should not be used.
      PIPELINE_RUN_CANCELLED: Cancelled status.
    """
        PIPELINE_RUN_STATUS_UNSPECIFIED = 0
        PIPELINE_RUN_CANCELLED = 1

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
    childReferences = _messages.MessageField('ChildStatusReference', 2, repeated=True)
    completionTime = _messages.StringField(3)
    conditions = _messages.MessageField('GoogleDevtoolsCloudbuildV2Condition', 4, repeated=True)
    createTime = _messages.StringField(5)
    etag = _messages.StringField(6)
    finallyStartTime = _messages.StringField(7)
    gcbParams = _messages.MessageField('GcbParamsValue', 8)
    name = _messages.StringField(9)
    params = _messages.MessageField('Param', 10, repeated=True)
    pipelineRef = _messages.MessageField('PipelineRef', 11)
    pipelineRunStatus = _messages.EnumField('PipelineRunStatusValueValuesEnum', 12)
    pipelineSpec = _messages.MessageField('PipelineSpec', 13)
    provenance = _messages.MessageField('Provenance', 14)
    record = _messages.StringField(15)
    resolvedPipelineSpec = _messages.MessageField('PipelineSpec', 16)
    results = _messages.MessageField('PipelineRunResult', 17, repeated=True)
    security = _messages.MessageField('Security', 18)
    serviceAccount = _messages.StringField(19)
    skippedTasks = _messages.MessageField('SkippedTask', 20, repeated=True)
    startTime = _messages.StringField(21)
    timeouts = _messages.MessageField('TimeoutFields', 22)
    uid = _messages.StringField(23)
    updateTime = _messages.StringField(24)
    worker = _messages.MessageField('Worker', 25)
    workerPool = _messages.StringField(26)
    workflow = _messages.StringField(27)
    workspaces = _messages.MessageField('WorkspaceBinding', 28, repeated=True)