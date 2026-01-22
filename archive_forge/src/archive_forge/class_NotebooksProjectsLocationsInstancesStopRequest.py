from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesStopRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesStopRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    stopInstanceRequest: A StopInstanceRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    stopInstanceRequest = _messages.MessageField('StopInstanceRequest', 2)