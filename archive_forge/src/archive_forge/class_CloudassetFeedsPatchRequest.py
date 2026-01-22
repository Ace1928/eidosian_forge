from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetFeedsPatchRequest(_messages.Message):
    """A CloudassetFeedsPatchRequest object.

  Fields:
    name: Required. The format will be
      projects/{project_number}/feeds/{client-assigned_feed_identifier} or
      folders/{folder_number}/feeds/{client-assigned_feed_identifier} or
      organizations/{organization_number}/feeds/{client-
      assigned_feed_identifier} The client-assigned feed identifier must be
      unique within the parent project/folder/organization.
    updateFeedRequest: A UpdateFeedRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateFeedRequest = _messages.MessageField('UpdateFeedRequest', 2)