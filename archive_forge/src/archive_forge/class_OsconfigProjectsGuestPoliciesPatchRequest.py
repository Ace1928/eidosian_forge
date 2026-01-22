from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsGuestPoliciesPatchRequest(_messages.Message):
    """A OsconfigProjectsGuestPoliciesPatchRequest object.

  Fields:
    guestPolicy: A GuestPolicy resource to be passed as the request body.
    name: Required. Unique name of the resource in this project using one of
      the following forms:
      `projects/{project_number}/guestPolicies/{guest_policy_id}`.
    updateMask: Field mask that controls which fields of the guest policy
      should be updated.
  """
    guestPolicy = _messages.MessageField('GuestPolicy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)