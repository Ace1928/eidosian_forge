from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesRefreshRuntimeTokenInternalRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesRefreshRuntimeTokenInternalRequest
  object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtime_id}`
    refreshRuntimeTokenInternalRequest: A RefreshRuntimeTokenInternalRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    refreshRuntimeTokenInternalRequest = _messages.MessageField('RefreshRuntimeTokenInternalRequest', 2)