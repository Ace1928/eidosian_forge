from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2GCSVolumeSource(_messages.Message):
    """Represents a volume backed by a Cloud Storage bucket using Cloud Storage
  FUSE.

  Fields:
    bucket: Cloud Storage Bucket name.
    readOnly: If true, the volume will be mounted as read only for all mounts.
  """
    bucket = _messages.StringField(1)
    readOnly = _messages.BooleanField(2)