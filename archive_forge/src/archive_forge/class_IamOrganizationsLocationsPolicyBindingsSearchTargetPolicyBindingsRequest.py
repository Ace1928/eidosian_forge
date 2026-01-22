from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPolicyBindingsSearchTargetPolicyBindingsRequest(_messages.Message):
    """A
  IamOrganizationsLocationsPolicyBindingsSearchTargetPolicyBindingsRequest
  object.

  Fields:
    pageSize: Optional. The maximum number of policy bindings to return. The
      service may return fewer than this value. If unspecified, at most 50
      policy bindings will be returned. The maximum value is 1000; values
      above 1000 will be coerced to 1000.
    pageToken: Optional. A page token, received from a previous
      `SearchTargetPolicyBindingsRequest` call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `SearchTargetPolicyBindingsRequest` must match the call that provided
      the page token.
    parent: Required. The parent resource where this search will be performed.
      This should be the nearest CRM resource to the target. Format:
      `projects/{project_id}/locations/{location}`
      `projects/{project_number}/locations/{location}`
      `folders/{folder_id}/locations/{location}`
      `organizations/{organization_id}/locations/{location}`
    target: Required. The target resource, which is bound to the policy in the
      binding. Format:
      `//iam.googleapis.com/locations/global/workforcePools/POOL_ID` `//iam.go
      ogleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPo
      ols/POOL_ID`
      `//iam.googleapis.com/locations/global/workspace/WORKSPACE_ID`
      `//cloudresourcemanager.googleapis.com/projects/{project_number}`
      `//cloudresourcemanager.googleapis.com/folders/{folder_id}`
      `//cloudresourcemanager.googleapis.com/organizations/{organization_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    target = _messages.StringField(4)