from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCreateRequest(_messages.Message):
    """A WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCreateRequest
  object.

  Fields:
    execution: A Execution resource to be passed as the request body.
    parent: Required. Name of the workflow for which an execution should be
      created. Format:
      projects/{project}/locations/{location}/workflows/{workflow} The latest
      revision of the workflow will be used.
  """
    execution = _messages.MessageField('Execution', 1)
    parent = _messages.StringField(2, required=True)