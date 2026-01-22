from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Session(_messages.Message):
    """Session carries the debug session id and its creation time.

  Fields:
    id: The debug session ID.
    timestampMs: The first transaction creation timestamp in millisecond,
      recorded by UAP.
  """
    id = _messages.StringField(1)
    timestampMs = _messages.IntegerField(2)