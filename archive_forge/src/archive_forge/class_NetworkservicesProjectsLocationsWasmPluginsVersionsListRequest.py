from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmPluginsVersionsListRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmPluginsVersionsListRequest object.

  Fields:
    pageSize: Maximum number of `WasmPluginVersion` resources to return per
      call. If not specified, at most 50 `WasmPluginVersion`s are returned.
      The maximum value is 1000; values above 1000 are coerced to 1000.
    pageToken: The value returned by the last `ListWasmPluginVersionsResponse`
      call. Indicates that this is a continuation of a prior
      `ListWasmPluginVersions` call, and that the next page of data is to be
      returned.
    parent: Required. The `WasmPlugin` resource whose `WasmPluginVersion`s are
      listed, specified in the following format:
      `projects/{project}/locations/global/wasmPlugins/{wasm_plugin}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)