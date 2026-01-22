from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowsProjectsLocationsWorkflowsListRequest(_messages.Message):
    """A WorkflowsProjectsLocationsWorkflowsListRequest object.

  Fields:
    filter: Filter to restrict results to specific workflows.
    orderBy: Comma-separated list of fields that that specify the order of the
      results. Default sorting order for a field is ascending. To specify
      descending order for a field, append a " desc" suffix. If not specified,
      the results will be returned in an unspecified order.
    pageSize: Maximum number of workflows to return per call. The service may
      return fewer than this value. If the value is not specified, a default
      value of 500 will be used. The maximum permitted value is 1000 and
      values greater than 1000 will be coerced down to 1000.
    pageToken: A page token, received from a previous `ListWorkflows` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListWorkflows` must match the call that provided
      the page token.
    parent: Required. Project and location from which the workflows should be
      listed. Format: projects/{project}/locations/{location}
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)