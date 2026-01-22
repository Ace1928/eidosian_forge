from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListInstanceAttachmentsResponse(_messages.Message):
    """Response for ListInstanceAttachments.

  Fields:
    attachments: Attachments for the instance.
    nextPageToken: Page token that you can include in a
      ListInstanceAttachments request to retrieve the next page of content. If
      omitted, no subsequent pages exist.
  """
    attachments = _messages.MessageField('GoogleCloudApigeeV1InstanceAttachment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)