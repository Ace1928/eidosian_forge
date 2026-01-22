from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageSource(_messages.Message):
    """StorageSource describes the location of the source in an archive file in
  Google Cloud Storage.

  Fields:
    bucket: Google Cloud Storage bucket containing source (see [Bucket Name
      Requirements] (https://cloud.google.com/storage/docs/bucket-
      naming#requirements)).
    generation: Google Cloud Storage generation for the object.
    object: Google Cloud Storage object containing source.
  """
    bucket = _messages.StringField(1)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3)