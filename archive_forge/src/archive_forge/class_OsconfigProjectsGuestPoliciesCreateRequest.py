from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsGuestPoliciesCreateRequest(_messages.Message):
    """A OsconfigProjectsGuestPoliciesCreateRequest object.

  Fields:
    guestPolicy: A GuestPolicy resource to be passed as the request body.
    guestPolicyId: Required. The logical name of the guest policy in the
      project with the following restrictions: * Must contain only lowercase
      letters, numbers, and hyphens. * Must start with a letter. * Must be
      between 1-63 characters. * Must end with a number or a letter. * Must be
      unique within the project.
    parent: Required. The resource name of the parent using one of the
      following forms: `projects/{project_number}`.
  """
    guestPolicy = _messages.MessageField('GuestPolicy', 1)
    guestPolicyId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)