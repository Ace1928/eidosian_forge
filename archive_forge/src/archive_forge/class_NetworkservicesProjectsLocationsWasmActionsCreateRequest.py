from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmActionsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmActionsCreateRequest object.

  Fields:
    parent: Required. The parent resource of the `WasmAction` resource. Must
      be in the format `projects/{project}/locations/global`.
    wasmAction: A WasmAction resource to be passed as the request body.
    wasmActionId: Required. User-provided ID of the `WasmAction` resource to
      be created.
  """
    parent = _messages.StringField(1, required=True)
    wasmAction = _messages.MessageField('WasmAction', 2)
    wasmActionId = _messages.StringField(3)