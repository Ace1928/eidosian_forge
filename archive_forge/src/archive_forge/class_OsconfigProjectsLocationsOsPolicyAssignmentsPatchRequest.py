from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsOsPolicyAssignmentsPatchRequest(_messages.Message):
    """A OsconfigProjectsLocationsOsPolicyAssignmentsPatchRequest object.

  Fields:
    name: Resource name. Format: `projects/{project_number}/locations/{locatio
      n}/osPolicyAssignments/{os_policy_assignment_id}` This field is ignored
      when you create an OS policy assignment.
    oSPolicyAssignment: A OSPolicyAssignment resource to be passed as the
      request body.
    updateMask: Optional. Field mask that controls which fields of the
      assignment should be updated.
  """
    name = _messages.StringField(1, required=True)
    oSPolicyAssignment = _messages.MessageField('OSPolicyAssignment', 2)
    updateMask = _messages.StringField(3)