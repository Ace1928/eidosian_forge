from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
def GetConnectionSchemaMetadata(self, request, global_params=None):
    """Gets schema metadata of a connection. SchemaMetadata is a singleton resource for each connection.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsGetConnectionSchemaMetadataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectionSchemaMetadata) The response message.
      """
    config = self.GetMethodConfig('GetConnectionSchemaMetadata')
    return self._RunMethod(config, request, global_params=global_params)