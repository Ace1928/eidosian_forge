from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesSwitchRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesSwitchRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtime_id}`
    switchRuntimeRequest: A SwitchRuntimeRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    switchRuntimeRequest = _messages.MessageField('SwitchRuntimeRequest', 2)