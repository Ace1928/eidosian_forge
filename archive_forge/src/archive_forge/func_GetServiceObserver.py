from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1alpha1 import networkservices_v1alpha1_messages as messages
def GetServiceObserver(self, request, global_params=None):
    """GetServiceObserver gets a singleton without a parent resource.

      Args:
        request: (NetworkservicesProjectsLocationsGlobalGetServiceObserverRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceObserver) The response message.
      """
    config = self.GetMethodConfig('GetServiceObserver')
    return self._RunMethod(config, request, global_params=global_params)