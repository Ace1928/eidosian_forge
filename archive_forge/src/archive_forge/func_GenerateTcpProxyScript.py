from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1 import datamigration_v1_messages as messages
def GenerateTcpProxyScript(self, request, global_params=None):
    """Generate a TCP Proxy configuration script to configure a cloud-hosted VM running a TCP Proxy.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsGenerateTcpProxyScriptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TcpProxyScript) The response message.
      """
    config = self.GetMethodConfig('GenerateTcpProxyScript')
    return self._RunMethod(config, request, global_params=global_params)