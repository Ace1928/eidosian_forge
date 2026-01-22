from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1ListOrgPolicyViolationsPreviewsResponse(_messages.Message):
    """ListOrgPolicyViolationsPreviewsResponse is the response message for
  OrgPolicyViolationsPreviewService.ListOrgPolicyViolationsPreviews.

  Fields:
    nextPageToken: A token that you can use to retrieve the next page of
      results. If this field is omitted, there are no subsequent pages.
    orgPolicyViolationsPreviews: The list of OrgPolicyViolationsPreview
  """
    nextPageToken = _messages.StringField(1)
    orgPolicyViolationsPreviews = _messages.MessageField('GoogleCloudPolicysimulatorV1OrgPolicyViolationsPreview', 2, repeated=True)