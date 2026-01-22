from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
def GetMetrics(self, request, global_params=None):
    """Get some aggregated metrics for a Key. This data can be used to build dashboards.

      Args:
        request: (RecaptchaenterpriseProjectsKeysGetMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Metrics) The response message.
      """
    config = self.GetMethodConfig('GetMetrics')
    return self._RunMethod(config, request, global_params=global_params)