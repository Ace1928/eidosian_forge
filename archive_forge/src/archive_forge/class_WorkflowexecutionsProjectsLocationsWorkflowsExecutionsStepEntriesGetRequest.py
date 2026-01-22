from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsStepEntriesGetRequest(_messages.Message):
    """A
  WorkflowexecutionsProjectsLocationsWorkflowsExecutionsStepEntriesGetRequest
  object.

  Fields:
    name: Required. The name of the step entry to retrieve. Format: projects/{
      project}/locations/{location}/workflows/{workflow}/executions/{execution
      }/stepEntries/{step_entry}
  """
    name = _messages.StringField(1, required=True)