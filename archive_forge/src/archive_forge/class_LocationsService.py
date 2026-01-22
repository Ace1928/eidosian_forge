from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class LocationsService(base_api.BaseApiService):
    """Service class for the locations resource."""
    _NAME = 'locations'

    def __init__(self, client):
        super(CloudbuildV1.LocationsService, self).__init__(client)
        self._upload_configs = {}

    def RegionalWebhook(self, request, global_params=None):
        """ReceiveRegionalWebhook is called when the API receives a regional GitHub webhook.

      Args:
        request: (CloudbuildLocationsRegionalWebhookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('RegionalWebhook')
        return self._RunMethod(config, request, global_params=global_params)
    RegionalWebhook.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/regionalWebhook', http_method='POST', method_id='cloudbuild.locations.regionalWebhook', ordered_params=['location'], path_params=['location'], query_params=['webhookKey'], relative_path='v1/{+location}/regionalWebhook', request_field='httpBody', request_type_name='CloudbuildLocationsRegionalWebhookRequest', response_type_name='Empty', supports_download=False)