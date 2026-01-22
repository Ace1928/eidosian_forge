from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsListRequest(_messages.Message):
    """A
  OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsListRequest
  object.

  Fields:
    filter: If provided, this field specifies the criteria that must be met by
      the `OSPolicyAssignmentReport` API resource that is included in the
      response.
    pageSize: The maximum number of results to return.
    pageToken: A pagination token returned from a previous call to the
      `ListOSPolicyAssignmentReports` method that indicates where this listing
      should continue from.
    parent: Required. The parent resource name. Format: `projects/{project}/lo
      cations/{location}/instances/{instance}/osPolicyAssignments/{assignment}
      /reports` For `{project}`, either `project-number` or `project-id` can
      be provided. For `{instance}`, either `instance-name`, `instance-id`, or
      `-` can be provided. If '-' is provided, the response will include
      OSPolicyAssignmentReports for all instances in the project/location. For
      `{assignment}`, either `assignment-id` or `-` can be provided. If '-' is
      provided, the response will include OSPolicyAssignmentReports for all
      OSPolicyAssignments in the project/location. Either {instance} or
      {assignment} must be `-`. For example: `projects/{project}/locations/{lo
      cation}/instances/{instance}/osPolicyAssignments/-/reports` returns all
      reports for the instance `projects/{project}/locations/{location}/instan
      ces/-/osPolicyAssignments/{assignment-id}/reports` returns all the
      reports for the given assignment across all instances. `projects/{projec
      t}/locations/{location}/instances/-/osPolicyAssignments/-/reports`
      returns all the reports for all assignments across all instances.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)