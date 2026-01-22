from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsResponse(_messages.Message):
    """Response for BatchComputeSecurityAssessmentResults.

  Fields:
    assessmentTime: The time of the assessment api call.
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is blank, there are no subsequent pages.
    securityAssessmentResults: Default sort order is by resource name in
      alphabetic order.
  """
    assessmentTime = _messages.StringField(1)
    nextPageToken = _messages.StringField(2)
    securityAssessmentResults = _messages.MessageField('GoogleCloudApigeeV1SecurityAssessmentResult', 3, repeated=True)