from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConsumerOverridesResponse(_messages.Message):
    """Response message for ListConsumerOverrides.

  Fields:
    nextPageToken: Token identifying which result to start with; returned by a
      previous list call.
    overrides: Consumer overrides on this limit.
  """
    nextPageToken = _messages.StringField(1)
    overrides = _messages.MessageField('QuotaOverride', 2, repeated=True)