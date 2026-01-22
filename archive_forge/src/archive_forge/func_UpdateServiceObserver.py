from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1alpha1 import networkservices_v1alpha1_messages as messages
def UpdateServiceObserver(self, request, global_params=None):
    """UpdateServiceObserver updates a singleton without a parent resource.

      Args:
        request: (NetworkservicesProjectsLocationsGlobalUpdateServiceObserverRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceObserver) The response message.
      """
    config = self.GetMethodConfig('UpdateServiceObserver')
    return self._RunMethod(config, request, global_params=global_params)