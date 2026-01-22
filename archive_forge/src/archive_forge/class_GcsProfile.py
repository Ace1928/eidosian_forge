from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsProfile(_messages.Message):
    """Cloud Storage bucket profile.

  Fields:
    bucket: Required. The Cloud Storage bucket name.
    rootPath: The root path inside the Cloud Storage bucket.
  """
    bucket = _messages.StringField(1)
    rootPath = _messages.StringField(2)