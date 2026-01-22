from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IngestionDataSourceSettings(_messages.Message):
    """Settings for an ingestion data source on a topic.

  Fields:
    awsKinesis: Optional. Amazon Kinesis Data Streams.
    cloudStorage: Optional. Cloud Storage.
  """
    awsKinesis = _messages.MessageField('AwsKinesis', 1)
    cloudStorage = _messages.MessageField('CloudStorage', 2)