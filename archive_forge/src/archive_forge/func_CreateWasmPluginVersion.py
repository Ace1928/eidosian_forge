from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.service_extensions import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
import six
def CreateWasmPluginVersion(self, name, parent, image, plugin_config_data=None, plugin_config_uri=None, description=None, labels=None):
    """Calls the CreateWasmPluginVersion API.

    Args:
      name: string, wasmPluginVersion's name.
      parent: string, wasmPluginVersion's parent relative name.
      image: string, URI of the container image containing the Wasm module,
        stored in the Artifact Registry.
      plugin_config_data: string or bytes, WasmPlugin configuration in the
        textual or binary format.
      plugin_config_uri: string, URI of the container image containing the
        plugin configuration, stored in the Artifact Registry.
      description: string, human-readable description of the service.
      labels: set of label tags.

    Returns:
      (Operation) The response message.
    """
    plugin_config_data_binary = None
    if plugin_config_data:
        plugin_config_data_binary = six.ensure_binary(plugin_config_data)
    request = self.messages.NetworkservicesProjectsLocationsWasmPluginsVersionsCreateRequest(parent=parent, wasmPluginVersionId=name, wasmPluginVersion=self.messages.WasmPluginVersion(imageUri=image, description=description, labels=labels, pluginConfigData=plugin_config_data_binary, pluginConfigUri=plugin_config_uri))
    return self._wasm_plugin_version_client.Create(request)