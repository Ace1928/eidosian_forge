from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsGuestPoliciesGetRequest(_messages.Message):
    """A OsconfigProjectsGuestPoliciesGetRequest object.

  Fields:
    name: Required. The resource name of the guest policy using one of the
      following forms:
      `projects/{project_number}/guestPolicies/{guest_policy_id}`.
  """
    name = _messages.StringField(1, required=True)