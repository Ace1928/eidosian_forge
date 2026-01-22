from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsGovernanceRulesSetIamPolicyRequest(_messages.Message):
    """A DataplexProjectsLocationsGovernanceRulesSetIamPolicyRequest object.

  Fields:
    googleIamV1SetIamPolicyRequest: A GoogleIamV1SetIamPolicyRequest resource
      to be passed as the request body.
    resource: REQUIRED: The resource for which the policy is being specified.
      See Resource names (https://cloud.google.com/apis/design/resource_names)
      for the appropriate value for this field.
  """
    googleIamV1SetIamPolicyRequest = _messages.MessageField('GoogleIamV1SetIamPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)