from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigOrganizationsGetIamPolicyRequest(_messages.Message):
    """A OsconfigOrganizationsGetIamPolicyRequest object.

  Fields:
    getIamPolicyRequest: A GetIamPolicyRequest resource to be passed as the
      request body.
    resource: REQUIRED: The resource for which the policy is being requested.
      See the operation documentation for the appropriate value for this
      field.
  """
    getIamPolicyRequest = _messages.MessageField('GetIamPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)