from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPolicyBindingsGetRequest(_messages.Message):
    """A IamOrganizationsLocationsPolicyBindingsGetRequest object.

  Fields:
    name: Required. The name of the policy binding to retrieve. Format: `proje
      cts/{project_id}/locations/{location}/policyBindings/{policy_binding_id}
      ` `projects/{project_number}/locations/{location}/policyBindings/{policy
      _binding_id}` `folders/{folder_id}/locations/{location}/policyBindings/{
      policy_binding_id}` `organizations/{organization_id}/locations/{location
      }/policyBindings/{policy_binding_id}`
  """
    name = _messages.StringField(1, required=True)