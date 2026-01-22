from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetBackupIndexDownloadUrlResponse(_messages.Message):
    """Response message for GetBackupIndexDownloadUrl.

  Fields:
    signedUrl: A string attribute.
  """
    signedUrl = _messages.StringField(1)