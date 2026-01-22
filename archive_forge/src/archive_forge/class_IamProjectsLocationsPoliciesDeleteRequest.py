from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsPoliciesDeleteRequest(_messages.Message):
    """A IamProjectsLocationsPoliciesDeleteRequest object.

  Fields:
    etag: Optional. The expected etag of the policy to delete. If an `etag` is
      provided, the `etag` value must match the value that is stored in IAM.
      If the values don't match, the request fails with ABORTED status. If an
      `etag` is not provided, the policy will be deleted regardless of the
      existing policy data.
    force: Optional. If set to true, the request will force the deletion of
      the Policy even if there are PolicyBindings that refer to the Policy. If
      policy bindings are referenced by the policy, these bindings will have
      no effect in policy evaluation, and will be automatically deleted later.
    name: Required. The name of the policy to delete. The name needs to follow
      formats below.
      `projects/{project_id}/locations/{location}/policies/{policy_id}`
      `projects/{project_number}/locations/{location}/policies/{policy_id}`
      `folders/{numeric_id}/locations/{location}/policies/{policy_id}`
      `organizations/{numeric_id}/locations/{location}/policies/{policy_id}`
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with an expected result, but no actual change is made.
  """
    etag = _messages.StringField(1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)