from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsInstanceOSPoliciesCompliancesListRequest(_messages.Message):
    """A OsconfigProjectsLocationsInstanceOSPoliciesCompliancesListRequest
  object.

  Fields:
    filter: If provided, this field specifies the criteria that must be met by
      a `InstanceOSPoliciesCompliance` API resource to be included in the
      response.
    pageSize: The maximum number of results to return.
    pageToken: A pagination token returned from a previous call to
      `ListInstanceOSPoliciesCompliances` that indicates where this listing
      should continue from.
    parent: Required. The parent resource name. Format:
      `projects/{project}/locations/{location}` For `{project}`, either
      Compute Engine project-number or project-id can be provided.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)