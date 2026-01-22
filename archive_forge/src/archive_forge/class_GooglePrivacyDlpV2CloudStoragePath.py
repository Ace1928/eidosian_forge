from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CloudStoragePath(_messages.Message):
    """Message representing a single file or path in Cloud Storage.

  Fields:
    path: A URL representing a file or path (no wildcards) in Cloud Storage.
      Example: `gs://[BUCKET_NAME]/dictionary.txt`
  """
    path = _messages.StringField(1)