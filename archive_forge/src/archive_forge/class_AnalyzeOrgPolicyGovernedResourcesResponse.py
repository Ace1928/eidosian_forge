from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeOrgPolicyGovernedResourcesResponse(_messages.Message):
    """The response message for AssetService.AnalyzeOrgPolicyGovernedResources.

  Fields:
    constraint: The definition of the constraint in the request.
    governedResources: The list of the analyzed governed resources.
    nextPageToken: The page token to fetch the next page for
      AnalyzeOrgPolicyGovernedResourcesResponse.governed_resources.
  """
    constraint = _messages.MessageField('AnalyzerOrgPolicyConstraint', 1)
    governedResources = _messages.MessageField('GoogleCloudAssetV1GovernedResource', 2, repeated=True)
    nextPageToken = _messages.StringField(3)