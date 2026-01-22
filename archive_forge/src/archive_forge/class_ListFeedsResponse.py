from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFeedsResponse(_messages.Message):
    """A ListFeedsResponse object.

  Fields:
    feeds: A list of feeds.
  """
    feeds = _messages.MessageField('Feed', 1, repeated=True)