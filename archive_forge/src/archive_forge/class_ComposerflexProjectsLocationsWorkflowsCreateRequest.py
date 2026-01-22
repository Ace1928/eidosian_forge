from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerflexProjectsLocationsWorkflowsCreateRequest(_messages.Message):
    """A ComposerflexProjectsLocationsWorkflowsCreateRequest object.

  Fields:
    parent: Parent resource of the workflow to create. The parent must be of
      the form "projects/{projectId}/locations/{locationId}".
    workflow: A Workflow resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    workflow = _messages.MessageField('Workflow', 2)