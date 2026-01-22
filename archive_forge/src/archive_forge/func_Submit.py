from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
def Submit(self, request, global_params=None):
    """Creates a new service config (version) for a managed service based on.
user-supplied configuration sources files (for example: OpenAPI
Specification). This method stores the source configurations as well as the
generated service config. It does NOT apply the service config to any
backend services.

Operation<response: SubmitConfigSourceResponse>

      Args:
        request: (ServicemanagementServicesConfigsSubmitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Submit')
    return self._RunMethod(config, request, global_params=global_params)