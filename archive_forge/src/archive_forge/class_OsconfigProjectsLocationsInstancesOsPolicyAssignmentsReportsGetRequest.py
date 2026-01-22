from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsGetRequest(_messages.Message):
    """A OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsGetRequest
  object.

  Fields:
    name: Required. API resource name for OS policy assignment report. Format:
      `/projects/{project}/locations/{location}/instances/{instance}/osPolicyA
      ssignments/{assignment}/report` For `{project}`, either `project-number`
      or `project-id` can be provided. For `{instance_id}`, either Compute
      Engine `instance-id` or `instance-name` can be provided. For
      `{assignment_id}`, the OSPolicyAssignment id must be provided.
  """
    name = _messages.StringField(1, required=True)