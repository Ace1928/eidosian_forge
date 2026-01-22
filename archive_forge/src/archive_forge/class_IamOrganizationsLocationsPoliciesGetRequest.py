from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPoliciesGetRequest(_messages.Message):
    """A IamOrganizationsLocationsPoliciesGetRequest object.

  Fields:
    name: Required. The resource name for the Policy to be retrieved. The name
      needs to follow formats below.
      `projects/{project_id}/locations/{location}/policies/{policy_id}`
      `projects/{project_number}/locations/{location}/policies/{policy_id}`
      `folders/{numeric_id}/locations/{location}/policies/{policy_id}`
      `organizations/{numeric_id}/locations/{location}/policies/{policy_id}`
  """
    name = _messages.StringField(1, required=True)