from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListJobTriggersResponse(_messages.Message):
    """Response message for ListJobTriggers.

  Fields:
    jobTriggers: List of triggeredJobs, up to page_size in
      ListJobTriggersRequest.
    nextPageToken: If the next page is available then this value is the next
      page token to be used in the following ListJobTriggers request.
  """
    jobTriggers = _messages.MessageField('GooglePrivacyDlpV2JobTrigger', 1, repeated=True)
    nextPageToken = _messages.StringField(2)