from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1beta1 import serviceusage_v1beta1_messages as messages
def ImportConsumerOverrides(self, request, global_params=None):
    """Create or update multiple consumer overrides atomically, all on the.
same consumer, but on many different metrics or limits.
The name field in the quota override message should not be set.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsImportConsumerOverridesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ImportConsumerOverrides')
    return self._RunMethod(config, request, global_params=global_params)