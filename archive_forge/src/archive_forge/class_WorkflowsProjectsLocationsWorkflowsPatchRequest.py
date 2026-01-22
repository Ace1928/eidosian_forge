from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowsProjectsLocationsWorkflowsPatchRequest(_messages.Message):
    """A WorkflowsProjectsLocationsWorkflowsPatchRequest object.

  Fields:
    name: The resource name of the workflow. Format:
      projects/{project}/locations/{location}/workflows/{workflow}
    updateMask: List of fields to be updated. If not present, the entire
      workflow will be updated.
    workflow: A Workflow resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workflow = _messages.MessageField('Workflow', 3)