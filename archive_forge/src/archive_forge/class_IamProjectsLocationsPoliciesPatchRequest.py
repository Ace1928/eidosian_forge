from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsPoliciesPatchRequest(_messages.Message):
    """A IamProjectsLocationsPoliciesPatchRequest object.

  Fields:
    googleIamV3betaV3Policy: A GoogleIamV3betaV3Policy resource to be passed
      as the request body.
    name: The resource name of the `Policy`, which must be globally unique.
      The name needs to follow formats below. This field is output_only in a
      CreatePolicyRequest. Only `global` location is supported.
      `projects/{project_id}/locations/{location}/policies/{policy_id}`
      `projects/{project_number}/locations/{location}/policies/{policy_id}`
      `folders/{numeric_id}/locations/{location}/policies/{policy_id}`
      `organizations/{numeric_id}/locations/{location}/policies/{policy_id}`
    updateMask: Optional. The fields to update.
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with expected result, but no actual change is made.
  """
    googleIamV3betaV3Policy = _messages.MessageField('GoogleIamV3betaV3Policy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)