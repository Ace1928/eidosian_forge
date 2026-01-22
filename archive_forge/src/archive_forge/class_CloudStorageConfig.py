from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudStorageConfig(_messages.Message):
    """Configuration for a Cloud Storage subscription.

  Enums:
    StateValueValuesEnum: Output only. An output-only field that indicates
      whether or not the subscription can receive messages.

  Fields:
    avroConfig: Optional. If set, message data will be written to Cloud
      Storage in Avro format.
    bucket: Required. User-provided name for the Cloud Storage bucket. The
      bucket must be created by the user. The bucket name must be without any
      prefix like "gs://". See the [bucket naming requirements]
      (https://cloud.google.com/storage/docs/buckets#naming).
    filenameDatetimeFormat: Optional. User-provided format string specifying
      how to represent datetimes in Cloud Storage filenames. See the [datetime
      format guidance](https://cloud.google.com/pubsub/docs/create-
      cloudstorage-subscription#file_names).
    filenamePrefix: Optional. User-provided prefix for Cloud Storage filename.
      See the [object naming
      requirements](https://cloud.google.com/storage/docs/objects#naming).
    filenameSuffix: Optional. User-provided suffix for Cloud Storage filename.
      See the [object naming
      requirements](https://cloud.google.com/storage/docs/objects#naming).
      Must not end in "/".
    maxBytes: Optional. The maximum bytes that can be written to a Cloud
      Storage file before a new file is created. Min 1 KB, max 10 GiB. The
      max_bytes limit may be exceeded in cases where messages are larger than
      the limit.
    maxDuration: Optional. The maximum duration that can elapse before a new
      Cloud Storage file is created. Min 1 minute, max 10 minutes, default 5
      minutes. May not exceed the subscription's acknowledgement deadline.
    maxMessages: Optional. The maximum number of messages that can be written
      to a Cloud Storage file before a new file is created. Min 1000 messages.
    serviceAccountEmail: Optional. The service account to use to write to
      Cloud Storage. The subscription creator or updater that specifies this
      field must have `iam.serviceAccounts.actAs` permission on the service
      account. If not specified, the Pub/Sub [service
      agent](https://cloud.google.com/iam/docs/service-agents),
      service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com, is used.
    state: Output only. An output-only field that indicates whether or not the
      subscription can receive messages.
    textConfig: Optional. If set, message data will be written to Cloud
      Storage in text format.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. An output-only field that indicates whether or not the
    subscription can receive messages.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: The subscription can actively send messages to Cloud Storage.
      PERMISSION_DENIED: Cannot write to the Cloud Storage bucket because of
        permission denied errors.
      NOT_FOUND: Cannot write to the Cloud Storage bucket because it does not
        exist.
      IN_TRANSIT_LOCATION_RESTRICTION: Cannot write to the destination because
        enforce_in_transit is set to true and the destination locations are
        not in the allowed regions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PERMISSION_DENIED = 2
        NOT_FOUND = 3
        IN_TRANSIT_LOCATION_RESTRICTION = 4
    avroConfig = _messages.MessageField('AvroConfig', 1)
    bucket = _messages.StringField(2)
    filenameDatetimeFormat = _messages.StringField(3)
    filenamePrefix = _messages.StringField(4)
    filenameSuffix = _messages.StringField(5)
    maxBytes = _messages.IntegerField(6)
    maxDuration = _messages.StringField(7)
    maxMessages = _messages.IntegerField(8)
    serviceAccountEmail = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    textConfig = _messages.MessageField('TextConfig', 11)