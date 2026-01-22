from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GcsObject(_messages.Message):
    """Google Cloud Storage object representation.

  Fields:
    bucket: Required. Bucket of the Google Cloud Storage object.
    generationNumber: Required. Generation number of the Google Cloud Storage
      object. This is used to ensure that the ExecStep specified by this
      PatchJob does not change.
    object: Required. Name of the Google Cloud Storage object.
  """
    bucket = _messages.StringField(1)
    generationNumber = _messages.IntegerField(2)
    object = _messages.StringField(3)