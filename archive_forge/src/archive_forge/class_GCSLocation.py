from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GCSLocation(_messages.Message):
    """Represents a storage location in Cloud Storage

  Fields:
    bucket: Cloud Storage bucket. See
      https://cloud.google.com/storage/docs/naming#requirements
    generation: Cloud Storage generation for the object. If the generation is
      omitted, the latest generation will be used.
    object: Cloud Storage object. See
      https://cloud.google.com/storage/docs/naming#objectnames
  """
    bucket = _messages.StringField(1)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3)