from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsListRequest(_messages.Message):
    """A WorkflowexecutionsProjectsLocationsWorkflowsExecutionsListRequest
  object.

  Enums:
    ViewValueValuesEnum: Optional. A view defining which fields should be
      filled in the returned executions. The API will default to the BASIC
      view.

  Fields:
    pageSize: Maximum number of executions to return per call. Max supported
      value depends on the selected Execution view: it's 10000 for BASIC and
      100 for FULL. The default value used if the field is not specified is
      100, regardless of the selected view. Values greater than the max value
      will be coerced down to it.
    pageToken: A page token, received from a previous `ListExecutions` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListExecutions` must match the call that
      provided the page token.
    parent: Required. Name of the workflow for which the executions should be
      listed. Format:
      projects/{project}/locations/{location}/workflows/{workflow}
    view: Optional. A view defining which fields should be filled in the
      returned executions. The API will default to the BASIC view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. A view defining which fields should be filled in the
    returned executions. The API will default to the BASIC view.

    Values:
      EXECUTION_VIEW_UNSPECIFIED: The default / unset value.
      BASIC: Includes only basic metadata about the execution. Following
        fields are returned: name, start_time, end_time, state and
        workflow_revision_id.
      FULL: Includes all data.
    """
        EXECUTION_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)