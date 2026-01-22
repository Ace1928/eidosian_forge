from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTagKeysResponse(_messages.Message):
    """The ListTagKeys response message.

  Fields:
    nextPageToken: A pagination token returned from a previous call to
      `ListTagKeys` that indicates from where listing should continue.
    tagKeys: List of TagKeys that live under the specified parent in the
      request.
  """
    nextPageToken = _messages.StringField(1)
    tagKeys = _messages.MessageField('TagKey', 2, repeated=True)