from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyAssignmentReport(_messages.Message):
    """A report of the OS policy assignment status for a given instance.

  Fields:
    instance: The Compute Engine VM instance name.
    lastRunId: Unique identifier of the last attempted run to apply the OS
      policies associated with this assignment on the VM. This ID is logged by
      the OS Config agent while applying the OS policies associated with this
      assignment on the VM. NOTE: If the service is unable to successfully
      connect to the agent for this run, then this id will not be available in
      the agent logs.
    name: The `OSPolicyAssignmentReport` API resource name. Format: `projects/
      {project_number}/locations/{location}/instances/{instance_id}/osPolicyAs
      signments/{os_policy_assignment_id}/report`
    osPolicyAssignment: Reference to the `OSPolicyAssignment` API resource
      that the `OSPolicy` belongs to. Format: `projects/{project_number}/locat
      ions/{location}/osPolicyAssignments/{os_policy_assignment_id@revision_id
      }`
    osPolicyCompliances: Compliance data for each `OSPolicy` that is applied
      to the VM.
    updateTime: Timestamp for when the report was last generated.
  """
    instance = _messages.StringField(1)
    lastRunId = _messages.StringField(2)
    name = _messages.StringField(3)
    osPolicyAssignment = _messages.StringField(4)
    osPolicyCompliances = _messages.MessageField('OSPolicyAssignmentReportOSPolicyCompliance', 5, repeated=True)
    updateTime = _messages.StringField(6)