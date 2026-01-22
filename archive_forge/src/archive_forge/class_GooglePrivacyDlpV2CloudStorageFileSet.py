from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CloudStorageFileSet(_messages.Message):
    """Message representing a set of files in Cloud Storage.

  Fields:
    url: The url, in the format `gs:///`. Trailing wildcard in the path is
      allowed.
  """
    url = _messages.StringField(1)