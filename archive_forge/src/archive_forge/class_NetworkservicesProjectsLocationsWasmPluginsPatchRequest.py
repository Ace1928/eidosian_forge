from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmPluginsPatchRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmPluginsPatchRequest object.

  Fields:
    name: Required. Name of the `WasmPlugin` resource in the following format:
      `projects/{project}/locations/{location}/wasmPlugins/{wasm_plugin}`.
    updateMask: Optional. Used to specify the fields to be overwritten in the
      `WasmPlugin` resource by the update. The fields specified in the
      `update_mask` field are relative to the resource, not the full request.
      An omitted `update_mask` field is treated as an implied `update_mask`
      field equivalent to all fields that are populated (that have a non-empty
      value). The `update_mask` field supports a special value `*`, which
      means that each field in the given `WasmPlugin` resource (including the
      empty ones) replaces the current value.
    wasmPlugin: A WasmPlugin resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    wasmPlugin = _messages.MessageField('WasmPlugin', 3)