from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1Pipeline(_messages.Message):
    """The main pipeline entity and all the necessary metadata for launching
  and managing linked jobs.

  Enums:
    StateValueValuesEnum: Required. The state of the pipeline. When the
      pipeline is created, the state is set to 'PIPELINE_STATE_ACTIVE' by
      default. State changes can be requested by setting the state to
      stopping, paused, or resuming. State cannot be changed through
      UpdatePipeline requests.
    TypeValueValuesEnum: Required. The type of the pipeline. This field
      affects the scheduling of the pipeline and the type of metrics to show
      for the pipeline.

  Messages:
    PipelineSourcesValue: Immutable. The sources of the pipeline (for example,
      Dataplex). The keys and values are set by the corresponding sources
      during pipeline creation.

  Fields:
    createTime: Output only. Immutable. The timestamp when the pipeline was
      initially created. Set by the Data Pipelines service.
    displayName: Required. The display name of the pipeline. It can contain
      only letters ([A-Za-z]), numbers ([0-9]), hyphens (-), and underscores
      (_).
    jobCount: Output only. Number of jobs.
    lastUpdateTime: Output only. Immutable. The timestamp when the pipeline
      was last modified. Set by the Data Pipelines service.
    name: The pipeline name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/pipelines/PIPELINE_ID`. *
      `PROJECT_ID` can contain letters ([A-Za-z]), numbers ([0-9]), hyphens
      (-), colons (:), and periods (.). For more information, see [Identifying
      projects](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects#identifying_projects). * `LOCATION_ID` is the
      canonical ID for the pipeline's location. The list of available
      locations can be obtained by calling
      `google.cloud.location.Locations.ListLocations`. Note that the Data
      Pipelines service is not available in all regions. It depends on Cloud
      Scheduler, an App Engine application, so it's only available in [App
      Engine regions](https://cloud.google.com/about/locations#region). *
      `PIPELINE_ID` is the ID of the pipeline. Must be unique for the selected
      project and location.
    pipelineSources: Immutable. The sources of the pipeline (for example,
      Dataplex). The keys and values are set by the corresponding sources
      during pipeline creation.
    scheduleInfo: Internal scheduling information for a pipeline. If this
      information is provided, periodic jobs will be created per the schedule.
      If not, users are responsible for creating jobs externally.
    schedulerServiceAccountEmail: Optional. A service account email to be used
      with the Cloud Scheduler job. If not specified, the default compute
      engine service account will be used.
    state: Required. The state of the pipeline. When the pipeline is created,
      the state is set to 'PIPELINE_STATE_ACTIVE' by default. State changes
      can be requested by setting the state to stopping, paused, or resuming.
      State cannot be changed through UpdatePipeline requests.
    type: Required. The type of the pipeline. This field affects the
      scheduling of the pipeline and the type of metrics to show for the
      pipeline.
    workload: Workload information for creating new jobs.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. The state of the pipeline. When the pipeline is created, the
    state is set to 'PIPELINE_STATE_ACTIVE' by default. State changes can be
    requested by setting the state to stopping, paused, or resuming. State
    cannot be changed through UpdatePipeline requests.

    Values:
      STATE_UNSPECIFIED: The pipeline state isn't specified.
      STATE_RESUMING: The pipeline is getting started or resumed. When
        finished, the pipeline state will be 'PIPELINE_STATE_ACTIVE'.
      STATE_ACTIVE: The pipeline is actively running.
      STATE_STOPPING: The pipeline is in the process of stopping. When
        finished, the pipeline state will be 'PIPELINE_STATE_ARCHIVED'.
      STATE_ARCHIVED: The pipeline has been stopped. This is a terminal state
        and cannot be undone.
      STATE_PAUSED: The pipeline is paused. This is a non-terminal state. When
        the pipeline is paused, it will hold processing jobs, but can be
        resumed later. For a batch pipeline, this means pausing the scheduler
        job. For a streaming pipeline, creating a job snapshot to resume from
        will give the same effect.
    """
        STATE_UNSPECIFIED = 0
        STATE_RESUMING = 1
        STATE_ACTIVE = 2
        STATE_STOPPING = 3
        STATE_ARCHIVED = 4
        STATE_PAUSED = 5

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of the pipeline. This field affects the scheduling
    of the pipeline and the type of metrics to show for the pipeline.

    Values:
      PIPELINE_TYPE_UNSPECIFIED: The pipeline type isn't specified.
      PIPELINE_TYPE_BATCH: A batch pipeline. It runs jobs on a specific
        schedule, and each job will automatically terminate once execution is
        finished.
      PIPELINE_TYPE_STREAMING: A streaming pipeline. The underlying job is
        continuously running until it is manually terminated by the user. This
        type of pipeline doesn't have a schedule to run on, and the linked job
        gets created when the pipeline is created.
    """
        PIPELINE_TYPE_UNSPECIFIED = 0
        PIPELINE_TYPE_BATCH = 1
        PIPELINE_TYPE_STREAMING = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PipelineSourcesValue(_messages.Message):
        """Immutable. The sources of the pipeline (for example, Dataplex). The
    keys and values are set by the corresponding sources during pipeline
    creation.

    Messages:
      AdditionalProperty: An additional property for a PipelineSourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type PipelineSourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PipelineSourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    jobCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    lastUpdateTime = _messages.StringField(4)
    name = _messages.StringField(5)
    pipelineSources = _messages.MessageField('PipelineSourcesValue', 6)
    scheduleInfo = _messages.MessageField('GoogleCloudDatapipelinesV1ScheduleSpec', 7)
    schedulerServiceAccountEmail = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    type = _messages.EnumField('TypeValueValuesEnum', 10)
    workload = _messages.MessageField('GoogleCloudDatapipelinesV1Workload', 11)