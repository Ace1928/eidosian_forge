from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPoliciesGetRequest(_messages.Message):
    """A IamPoliciesGetRequest object.

  Fields:
    name: Required. The resource name of the policy to retrieve. Format:
      `policies/{attachment_point}/denypolicies/{policy_id}` Use the URL-
      encoded full resource name, which means that the forward-slash
      character, `/`, must be written as `%2F`. For example,
      `policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-
      project/denypolicies/my-policy`. For organizations and folders, use the
      numeric ID in the full resource name. For projects, you can use the
      alphanumeric or the numeric ID.
  """
    name = _messages.StringField(1, required=True)