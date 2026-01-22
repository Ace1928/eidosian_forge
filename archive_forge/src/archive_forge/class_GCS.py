from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GCS(_messages.Message):
    """Represents a Google Cloud Storage volume.

  Fields:
    remotePath: Remote path, either a bucket name or a subdirectory of a
      bucket, e.g.: bucket_name, bucket_name/subdirectory/
  """
    remotePath = _messages.StringField(1)