from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParamTypeValueValuesEnum(_messages.Enum):
    """Optional. The type of the parameter. Used for selecting input picker.

    Values:
      DEFAULT: Default input type.
      TEXT: The parameter specifies generic text input.
      GCS_READ_BUCKET: The parameter specifies a Cloud Storage Bucket to read
        from.
      GCS_WRITE_BUCKET: The parameter specifies a Cloud Storage Bucket to
        write to.
      GCS_READ_FILE: The parameter specifies a Cloud Storage file path to read
        from.
      GCS_WRITE_FILE: The parameter specifies a Cloud Storage file path to
        write to.
      GCS_READ_FOLDER: The parameter specifies a Cloud Storage folder path to
        read from.
      GCS_WRITE_FOLDER: The parameter specifies a Cloud Storage folder to
        write to.
      PUBSUB_TOPIC: The parameter specifies a Pub/Sub Topic.
      PUBSUB_SUBSCRIPTION: The parameter specifies a Pub/Sub Subscription.
      BIGQUERY_TABLE: The parameter specifies a BigQuery table.
      JAVASCRIPT_UDF_FILE: The parameter specifies a JavaScript UDF in Cloud
        Storage.
      SERVICE_ACCOUNT: The parameter specifies a Service Account email.
      MACHINE_TYPE: The parameter specifies a Machine Type.
      KMS_KEY_NAME: The parameter specifies a KMS Key name.
      WORKER_REGION: The parameter specifies a Worker Region.
      WORKER_ZONE: The parameter specifies a Worker Zone.
      BOOLEAN: The parameter specifies a boolean input.
      ENUM: The parameter specifies an enum input.
      NUMBER: The parameter specifies a number input.
    """
    DEFAULT = 0
    TEXT = 1
    GCS_READ_BUCKET = 2
    GCS_WRITE_BUCKET = 3
    GCS_READ_FILE = 4
    GCS_WRITE_FILE = 5
    GCS_READ_FOLDER = 6
    GCS_WRITE_FOLDER = 7
    PUBSUB_TOPIC = 8
    PUBSUB_SUBSCRIPTION = 9
    BIGQUERY_TABLE = 10
    JAVASCRIPT_UDF_FILE = 11
    SERVICE_ACCOUNT = 12
    MACHINE_TYPE = 13
    KMS_KEY_NAME = 14
    WORKER_REGION = 15
    WORKER_ZONE = 16
    BOOLEAN = 17
    ENUM = 18
    NUMBER = 19