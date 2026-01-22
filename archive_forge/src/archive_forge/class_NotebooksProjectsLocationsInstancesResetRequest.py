from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesResetRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesResetRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    resetInstanceRequest: A ResetInstanceRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    resetInstanceRequest = _messages.MessageField('ResetInstanceRequest', 2)