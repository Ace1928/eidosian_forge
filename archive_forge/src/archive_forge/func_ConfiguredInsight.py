from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
def ConfiguredInsight(self, request, global_params=None):
    """Gets the value for a selected particular insight based on the provided filters. Use the organization level path for fetching at org level and project level path for fetching the insight value specific to a particular project.

      Args:
        request: (BeyondcorpProjectsLocationsInsightsConfiguredInsightRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSaasplatformInsightsV1alphaConfiguredInsightResponse) The response message.
      """
    config = self.GetMethodConfig('ConfiguredInsight')
    return self._RunMethod(config, request, global_params=global_params)