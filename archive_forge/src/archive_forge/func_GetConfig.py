from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
def GetConfig(self, request, global_params=None):
    """Gets a service config (version) for a managed service. If `config_id` is.
not specified, the latest service config will be returned.

      Args:
        request: (ServicemanagementServicesGetConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
    config = self.GetMethodConfig('GetConfig')
    return self._RunMethod(config, request, global_params=global_params)