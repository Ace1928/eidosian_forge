from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
def GetDnsForwarding(self, request, global_params=None):
    """Gets details of the `DnsForwarding` config.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsGetDnsForwardingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsForwarding) The response message.
      """
    config = self.GetMethodConfig('GetDnsForwarding')
    return self._RunMethod(config, request, global_params=global_params)