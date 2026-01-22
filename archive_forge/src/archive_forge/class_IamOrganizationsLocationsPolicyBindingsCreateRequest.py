from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPolicyBindingsCreateRequest(_messages.Message):
    """A IamOrganizationsLocationsPolicyBindingsCreateRequest object.

  Fields:
    googleIamV3betaPolicyBinding: A GoogleIamV3betaPolicyBinding resource to
      be passed as the request body.
    parent: Required. The parent resource where this policy binding will be
      created. The binding parent is the closest CRM resource (i.e., Project,
      Folder or Organization) to the binding target. Format:
      `projects/{project_id}/locations/{location}`
      `projects/{project_number}/locations/{location}`
      `folders/{folder_id}/locations/{location}`
      `organizations/{organization_id}/locations/{location}`
    policyBindingId: Required. The ID to use for the policy binding, which
      will become the final component of the policy binding's resource name.
      This value must start with a lowercase letter followed by up to 62
      lowercase letters, numbers, hyphens, or dots. Pattern, /a-z{2,62}/.
    validateOnly: Optional. If set, validate the request and preview the
      creation, but do not actually post it.
  """
    googleIamV3betaPolicyBinding = _messages.MessageField('GoogleIamV3betaPolicyBinding', 1)
    parent = _messages.StringField(2, required=True)
    policyBindingId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)