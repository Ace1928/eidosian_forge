from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequestResourceArray(_messages.Message):
    """An array of resource messages.

  Fields:
    resources: Required. The array of resources. For Apigee, the proxies are
      resources.
  """
    resources = _messages.MessageField('GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequestResourceArrayResource', 1, repeated=True)