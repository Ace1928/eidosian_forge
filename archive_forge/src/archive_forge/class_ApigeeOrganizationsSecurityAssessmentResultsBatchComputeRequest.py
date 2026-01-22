from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityAssessmentResultsBatchComputeRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityAssessmentResultsBatchComputeRequest
  object.

  Fields:
    googleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequest: A
      GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequest resource
      to be passed as the request body.
    name: Required. Name of the organization for which the score needs to be
      computed in the following format:
      `organizations/{org}/securityAssessmentResults`
  """
    googleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequest = _messages.MessageField('GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequest', 1)
    name = _messages.StringField(2, required=True)