from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateFeedRequest(_messages.Message):
    """Update asset feed request.

  Fields:
    feed: Required. The new values of feed details. It must match an existing
      feed and the field `name` must be in the format of:
      projects/project_number/feeds/feed_id or
      folders/folder_number/feeds/feed_id or
      organizations/organization_number/feeds/feed_id.
    updateMask: Required. Only updates the `feed` fields indicated by this
      mask. The field mask must not be empty, and it must not contain fields
      that are immutable or only set by the server.
  """
    feed = _messages.MessageField('Feed', 1)
    updateMask = _messages.StringField(2)