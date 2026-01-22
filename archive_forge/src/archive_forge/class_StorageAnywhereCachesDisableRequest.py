from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageAnywhereCachesDisableRequest(_messages.Message):
    """A StorageAnywhereCachesDisableRequest object.

  Fields:
    anywhereCacheId: The ID of requested Anywhere Cache instance.
    bucket: Name of the parent bucket.
  """
    anywhereCacheId = _messages.StringField(1, required=True)
    bucket = _messages.StringField(2, required=True)