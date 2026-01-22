from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamProjectsServiceAccountsSetIamPolicyRequest(_messages.Message):
    """A IamProjectsServiceAccountsSetIamPolicyRequest object.

  Fields:
    resource: REQUIRED: The resource for which the policy is being specified.
      `resource` is usually specified as a path, such as
      `projects/*project*/zones/*zone*/disks/*disk*`.  The format for the path
      specified in this value is resource specific and is specified in the
      `setIamPolicy` documentation.
    setIamPolicyRequest: A SetIamPolicyRequest resource to be passed as the
      request body.
  """
    resource = _messages.StringField(1, required=True)
    setIamPolicyRequest = _messages.MessageField('SetIamPolicyRequest', 2)