from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListInspectTemplatesResponse(_messages.Message):
    """Response message for ListInspectTemplates.

  Fields:
    inspectTemplates: List of inspectTemplates, up to page_size in
      ListInspectTemplatesRequest.
    nextPageToken: If the next page is available then the next page token to
      be used in the following ListInspectTemplates request.
  """
    inspectTemplates = _messages.MessageField('GooglePrivacyDlpV2InspectTemplate', 1, repeated=True)
    nextPageToken = _messages.StringField(2)