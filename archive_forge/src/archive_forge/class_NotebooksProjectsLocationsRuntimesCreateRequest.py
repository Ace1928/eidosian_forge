from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesCreateRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesCreateRequest object.

  Fields:
    parent: Required. Format:
      `parent=projects/{project_id}/locations/{location}`
    requestId: Idempotent request UUID.
    runtime: A Runtime resource to be passed as the request body.
    runtimeId: Required. User-defined unique ID of this Runtime.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    runtime = _messages.MessageField('Runtime', 3)
    runtimeId = _messages.StringField(4)