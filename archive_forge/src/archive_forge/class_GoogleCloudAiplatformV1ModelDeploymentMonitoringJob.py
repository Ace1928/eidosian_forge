from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelDeploymentMonitoringJob(_messages.Message):
    """Represents a job that runs periodically to monitor the deployed models
  in an endpoint. It will analyze the logged training & prediction data to
  detect any abnormal behaviors.

  Enums:
    ScheduleStateValueValuesEnum: Output only. Schedule state when the
      monitoring job is in Running state.
    StateValueValuesEnum: Output only. The detailed state of the monitoring
      job. When the job is still creating, the state will be 'PENDING'. Once
      the job is successfully created, the state will be 'RUNNING'. Pause the
      job, the state will be 'PAUSED'. Resume the job, the state will return
      to 'RUNNING'.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      ModelDeploymentMonitoringJob. Label keys and values can be no longer
      than 64 characters (Unicode codepoints), can only contain lowercase
      letters, numeric characters, underscores and dashes. International
      characters are allowed. See https://goo.gl/xmQnxf for more information
      and examples of labels.

  Fields:
    analysisInstanceSchemaUri: YAML schema file uri describing the format of a
      single instance that you want Tensorflow Data Validation (TFDV) to
      analyze. If this field is empty, all the feature data types are inferred
      from predict_instance_schema_uri, meaning that TFDV will use the data in
      the exact format(data type) as prediction request/response. If there are
      any data type differences between predict instance and TFDV instance,
      this field can be used to override the schema. For models trained with
      Vertex AI, this field must be set as all the fields in predict instance
      formatted as string.
    bigqueryTables: Output only. The created bigquery tables for the job under
      customer project. Customer could do their own query & analysis. There
      could be 4 log tables in maximum: 1. Training data logging predict
      request/response 2. Serving data logging predict request/response
    createTime: Output only. Timestamp when this ModelDeploymentMonitoringJob
      was created.
    displayName: Required. The user-defined name of the
      ModelDeploymentMonitoringJob. The name can be up to 128 characters long
      and can consist of any UTF-8 characters. Display name of a
      ModelDeploymentMonitoringJob.
    enableMonitoringPipelineLogs: If true, the scheduled monitoring pipeline
      logs are sent to Google Cloud Logging, including pipeline status and
      anomalies detected. Please note the logs incur cost, which are subject
      to [Cloud Logging pricing](https://cloud.google.com/logging#pricing).
    encryptionSpec: Customer-managed encryption key spec for a
      ModelDeploymentMonitoringJob. If set, this ModelDeploymentMonitoringJob
      and all sub-resources of this ModelDeploymentMonitoringJob will be
      secured by this key.
    endpoint: Required. Endpoint resource name. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    error: Output only. Only populated when the job's state is
      `JOB_STATE_FAILED` or `JOB_STATE_CANCELLED`.
    labels: The labels with user-defined metadata to organize your
      ModelDeploymentMonitoringJob. Label keys and values can be no longer
      than 64 characters (Unicode codepoints), can only contain lowercase
      letters, numeric characters, underscores and dashes. International
      characters are allowed. See https://goo.gl/xmQnxf for more information
      and examples of labels.
    latestMonitoringPipelineMetadata: Output only. Latest triggered monitoring
      pipeline metadata.
    logTtl: The TTL of BigQuery tables in user projects which stores logs. A
      day is the basic unit of the TTL and we take the ceil of TTL/86400(a
      day). e.g. { second: 3600} indicates ttl = 1 day.
    loggingSamplingStrategy: Required. Sample Strategy for logging.
    modelDeploymentMonitoringObjectiveConfigs: Required. The config for
      monitoring objectives. This is a per DeployedModel config. Each
      DeployedModel needs to be configured separately.
    modelDeploymentMonitoringScheduleConfig: Required. Schedule config for
      running the monitoring job.
    modelMonitoringAlertConfig: Alert config for model monitoring.
    name: Output only. Resource name of a ModelDeploymentMonitoringJob.
    nextScheduleTime: Output only. Timestamp when this monitoring pipeline
      will be scheduled to run for the next round.
    predictInstanceSchemaUri: YAML schema file uri describing the format of a
      single instance, which are given to format this Endpoint's prediction
      (and explanation). If not set, we will generate predict schema from
      collected predict requests.
    samplePredictInstance: Sample Predict instance, same format as
      PredictRequest.instances, this can be set as a replacement of
      ModelDeploymentMonitoringJob.predict_instance_schema_uri. If not set, we
      will generate predict schema from collected predict requests.
    scheduleState: Output only. Schedule state when the monitoring job is in
      Running state.
    state: Output only. The detailed state of the monitoring job. When the job
      is still creating, the state will be 'PENDING'. Once the job is
      successfully created, the state will be 'RUNNING'. Pause the job, the
      state will be 'PAUSED'. Resume the job, the state will return to
      'RUNNING'.
    statsAnomaliesBaseDirectory: Stats anomalies base folder path.
    updateTime: Output only. Timestamp when this ModelDeploymentMonitoringJob
      was updated most recently.
  """

    class ScheduleStateValueValuesEnum(_messages.Enum):
        """Output only. Schedule state when the monitoring job is in Running
    state.

    Values:
      MONITORING_SCHEDULE_STATE_UNSPECIFIED: Unspecified state.
      PENDING: The pipeline is picked up and wait to run.
      OFFLINE: The pipeline is offline and will be scheduled for next run.
      RUNNING: The pipeline is running.
    """
        MONITORING_SCHEDULE_STATE_UNSPECIFIED = 0
        PENDING = 1
        OFFLINE = 2
        RUNNING = 3

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the monitoring job. When the job is
    still creating, the state will be 'PENDING'. Once the job is successfully
    created, the state will be 'RUNNING'. Pause the job, the state will be
    'PAUSED'. Resume the job, the state will return to 'RUNNING'.

    Values:
      JOB_STATE_UNSPECIFIED: The job state is unspecified.
      JOB_STATE_QUEUED: The job has been just created or resumed and
        processing has not yet begun.
      JOB_STATE_PENDING: The service is preparing to run the job.
      JOB_STATE_RUNNING: The job is in progress.
      JOB_STATE_SUCCEEDED: The job completed successfully.
      JOB_STATE_FAILED: The job failed.
      JOB_STATE_CANCELLING: The job is being cancelled. From this state the
        job may only go to either `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED` or
        `JOB_STATE_CANCELLED`.
      JOB_STATE_CANCELLED: The job has been cancelled.
      JOB_STATE_PAUSED: The job has been stopped, and can be resumed.
      JOB_STATE_EXPIRED: The job has expired.
      JOB_STATE_UPDATING: The job is being updated. Only jobs in the `RUNNING`
        state can be updated. After updating, the job goes back to the
        `RUNNING` state.
      JOB_STATE_PARTIALLY_SUCCEEDED: The job is partially succeeded, some
        results may be missing due to errors.
    """
        JOB_STATE_UNSPECIFIED = 0
        JOB_STATE_QUEUED = 1
        JOB_STATE_PENDING = 2
        JOB_STATE_RUNNING = 3
        JOB_STATE_SUCCEEDED = 4
        JOB_STATE_FAILED = 5
        JOB_STATE_CANCELLING = 6
        JOB_STATE_CANCELLED = 7
        JOB_STATE_PAUSED = 8
        JOB_STATE_EXPIRED = 9
        JOB_STATE_UPDATING = 10
        JOB_STATE_PARTIALLY_SUCCEEDED = 11

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your
    ModelDeploymentMonitoringJob. Label keys and values can be no longer than
    64 characters (Unicode codepoints), can only contain lowercase letters,
    numeric characters, underscores and dashes. International characters are
    allowed. See https://goo.gl/xmQnxf for more information and examples of
    labels.

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
    analysisInstanceSchemaUri = _messages.StringField(1)
    bigqueryTables = _messages.MessageField('GoogleCloudAiplatformV1ModelDeploymentMonitoringBigQueryTable', 2, repeated=True)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    enableMonitoringPipelineLogs = _messages.BooleanField(5)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 6)
    endpoint = _messages.StringField(7)
    error = _messages.MessageField('GoogleRpcStatus', 8)
    labels = _messages.MessageField('LabelsValue', 9)
    latestMonitoringPipelineMetadata = _messages.MessageField('GoogleCloudAiplatformV1ModelDeploymentMonitoringJobLatestMonitoringPipelineMetadata', 10)
    logTtl = _messages.StringField(11)
    loggingSamplingStrategy = _messages.MessageField('GoogleCloudAiplatformV1SamplingStrategy', 12)
    modelDeploymentMonitoringObjectiveConfigs = _messages.MessageField('GoogleCloudAiplatformV1ModelDeploymentMonitoringObjectiveConfig', 13, repeated=True)
    modelDeploymentMonitoringScheduleConfig = _messages.MessageField('GoogleCloudAiplatformV1ModelDeploymentMonitoringScheduleConfig', 14)
    modelMonitoringAlertConfig = _messages.MessageField('GoogleCloudAiplatformV1ModelMonitoringAlertConfig', 15)
    name = _messages.StringField(16)
    nextScheduleTime = _messages.StringField(17)
    predictInstanceSchemaUri = _messages.StringField(18)
    samplePredictInstance = _messages.MessageField('extra_types.JsonValue', 19)
    scheduleState = _messages.EnumField('ScheduleStateValueValuesEnum', 20)
    state = _messages.EnumField('StateValueValuesEnum', 21)
    statsAnomaliesBaseDirectory = _messages.MessageField('GoogleCloudAiplatformV1GcsDestination', 22)
    updateTime = _messages.StringField(23)