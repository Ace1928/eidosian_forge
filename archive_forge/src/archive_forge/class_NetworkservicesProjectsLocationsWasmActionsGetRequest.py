from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmActionsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmActionsGetRequest object.

  Fields:
    name: Required. A name of the `WasmAction` resource to get. Must be in the
      format `projects/{project}/locations/global/wasmActions/{wasm_action}`.
  """
    name = _messages.StringField(1, required=True)