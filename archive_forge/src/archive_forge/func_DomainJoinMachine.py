from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def DomainJoinMachine(self, request, global_params=None):
    """DomainJoinMachine API joins a Compute Engine VM to the domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsDomainJoinMachineRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainJoinMachineResponse) The response message.
      """
    config = self.GetMethodConfig('DomainJoinMachine')
    return self._RunMethod(config, request, global_params=global_params)