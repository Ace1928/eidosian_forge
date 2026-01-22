from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmPluginsVersionsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmPluginsVersionsGetRequest object.

  Fields:
    name: Required. A name of the `WasmPluginVersion` resource to get. Must be
      in the format `projects/{project}/locations/global/wasmPlugins/{wasm_plu
      gin}/versions/{wasm_plugin_version}`.
  """
    name = _messages.StringField(1, required=True)