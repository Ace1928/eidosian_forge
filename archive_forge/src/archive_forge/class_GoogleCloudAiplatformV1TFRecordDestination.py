from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1TFRecordDestination(_messages.Message):
    """The storage details for TFRecord output content.

  Fields:
    gcsDestination: Required. Google Cloud Storage location.
  """
    gcsDestination = _messages.MessageField('GoogleCloudAiplatformV1GcsDestination', 1)