from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferConfig(_messages.Message):
    """Represents a data transfer configuration. A transfer configuration
  contains all metadata needed to perform a data transfer. For example,
  `destination_dataset_id` specifies where data should be stored. When a new
  transfer configuration is created, the specified `destination_dataset_id` is
  created when needed and shared with the appropriate data source service
  account.

  Enums:
    StateValueValuesEnum: Output only. State of the most recently updated
      transfer run.

  Messages:
    ParamsValue: Parameters specific to each data source. For more information
      see the bq tab in the 'Setting up a data transfer' section for each data
      source. For example the parameters for Cloud Storage transfers are
      listed here: https://cloud.google.com/bigquery-transfer/docs/cloud-
      storage-transfer#bq

  Fields:
    dataRefreshWindowDays: The number of days to look back to automatically
      refresh the data. For example, if `data_refresh_window_days = 10`, then
      every day BigQuery reingests data for [today-10, today-1], rather than
      ingesting data for just [today-1]. Only valid if the data source
      supports the feature. Set the value to 0 to use the default value.
    dataSourceId: Data source ID. This cannot be changed once data transfer is
      created. The full list of available data source IDs can be returned
      through an API call: https://cloud.google.com/bigquery-transfer/docs/ref
      erence/datatransfer/rest/v1/projects.locations.dataSources/list
    datasetRegion: Output only. Region in which BigQuery dataset is located.
    destinationDatasetId: The BigQuery target dataset id.
    disabled: Is this config disabled. When set to true, no runs will be
      scheduled for this transfer config.
    displayName: User specified display name for the data transfer.
    emailPreferences: Email notifications will be sent according to these
      preferences to the email address of the user who owns this transfer
      config.
    encryptionConfiguration: The encryption configuration part. Currently, it
      is only used for the optional KMS key name. The BigQuery service account
      of your project must be granted permissions to use the key. Read methods
      will return the key name applied in effect. Write methods will apply the
      key if it is present, or otherwise try to apply project default keys if
      it is absent.
    name: Identifier. The resource name of the transfer config. Transfer
      config names have the form either
      `projects/{project_id}/locations/{region}/transferConfigs/{config_id}`
      or `projects/{project_id}/transferConfigs/{config_id}`, where
      `config_id` is usually a UUID, even though it is not guaranteed or
      required. The name is ignored when creating a transfer config.
    nextRunTime: Output only. Next time when data transfer will run.
    notificationPubsubTopic: Pub/Sub topic where notifications will be sent
      after transfer runs associated with this transfer config finish. The
      format for specifying a pubsub topic is:
      `projects/{project_id}/topics/{topic_id}`
    ownerInfo: Output only. Information about the user whose credentials are
      used to transfer data. Populated only for `transferConfigs.get`
      requests. In case the user information is not available, this field will
      not be populated.
    params: Parameters specific to each data source. For more information see
      the bq tab in the 'Setting up a data transfer' section for each data
      source. For example the parameters for Cloud Storage transfers are
      listed here: https://cloud.google.com/bigquery-transfer/docs/cloud-
      storage-transfer#bq
    schedule: Data transfer schedule. If the data source does not support a
      custom schedule, this should be empty. If it is empty, the default value
      for the data source will be used. The specified times are in UTC.
      Examples of valid format: `1st,3rd monday of month 15:30`, `every
      wed,fri of jan,jun 13:15`, and `first sunday of quarter 00:00`. See more
      explanation about the format here:
      https://cloud.google.com/appengine/docs/flexible/python/scheduling-jobs-
      with-cron-yaml#the_schedule_format NOTE: The minimum interval time
      between recurring transfers depends on the data source; refer to the
      documentation for your data source.
    scheduleOptions: Options customizing the data transfer schedule.
    state: Output only. State of the most recently updated transfer run.
    updateTime: Output only. Data transfer modification time. Ignored by
      server on input.
    userId: Deprecated. Unique ID of the user on whose behalf transfer is
      done.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the most recently updated transfer run.

    Values:
      TRANSFER_STATE_UNSPECIFIED: State placeholder (0).
      PENDING: Data transfer is scheduled and is waiting to be picked up by
        data transfer backend (2).
      RUNNING: Data transfer is in progress (3).
      SUCCEEDED: Data transfer completed successfully (4).
      FAILED: Data transfer failed (5).
      CANCELLED: Data transfer is cancelled (6).
    """
        TRANSFER_STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4
        CANCELLED = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParamsValue(_messages.Message):
        """Parameters specific to each data source. For more information see the
    bq tab in the 'Setting up a data transfer' section for each data source.
    For example the parameters for Cloud Storage transfers are listed here:
    https://cloud.google.com/bigquery-transfer/docs/cloud-storage-transfer#bq

    Messages:
      AdditionalProperty: An additional property for a ParamsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dataRefreshWindowDays = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    dataSourceId = _messages.StringField(2)
    datasetRegion = _messages.StringField(3)
    destinationDatasetId = _messages.StringField(4)
    disabled = _messages.BooleanField(5)
    displayName = _messages.StringField(6)
    emailPreferences = _messages.MessageField('EmailPreferences', 7)
    encryptionConfiguration = _messages.MessageField('EncryptionConfiguration', 8)
    name = _messages.StringField(9)
    nextRunTime = _messages.StringField(10)
    notificationPubsubTopic = _messages.StringField(11)
    ownerInfo = _messages.MessageField('UserInfo', 12)
    params = _messages.MessageField('ParamsValue', 13)
    schedule = _messages.StringField(14)
    scheduleOptions = _messages.MessageField('ScheduleOptions', 15)
    state = _messages.EnumField('StateValueValuesEnum', 16)
    updateTime = _messages.StringField(17)
    userId = _messages.IntegerField(18)