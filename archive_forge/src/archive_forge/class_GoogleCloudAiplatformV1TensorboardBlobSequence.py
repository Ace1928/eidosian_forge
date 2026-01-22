from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1TensorboardBlobSequence(_messages.Message):
    """One point viewable on a blob metric plot, but mostly just a wrapper
  message to work around repeated fields can't be used directly within `oneof`
  fields.

  Fields:
    values: List of blobs contained within the sequence.
  """
    values = _messages.MessageField('GoogleCloudAiplatformV1TensorboardBlob', 1, repeated=True)