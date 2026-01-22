from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class PublishersModelsService(base_api.BaseApiService):
    """Service class for the publishers_models resource."""
    _NAME = 'publishers_models'

    def __init__(self, client):
        super(AiplatformV1beta1.PublishersModelsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a Model Garden publisher model.

      Args:
        request: (AiplatformPublishersModelsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1PublisherModel) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/publishers/{publishersId}/models/{modelsId}', http_method='GET', method_id='aiplatform.publishers.models.get', ordered_params=['name'], path_params=['name'], query_params=['languageCode', 'view'], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformPublishersModelsGetRequest', response_type_name='GoogleCloudAiplatformV1beta1PublisherModel', supports_download=False)

    def List(self, request, global_params=None):
        """Lists publisher models in Model Garden.

      Args:
        request: (AiplatformPublishersModelsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ListPublisherModelsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/publishers/{publishersId}/models', http_method='GET', method_id='aiplatform.publishers.models.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'languageCode', 'orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1beta1/{+parent}/models', request_field='', request_type_name='AiplatformPublishersModelsListRequest', response_type_name='GoogleCloudAiplatformV1beta1ListPublisherModelsResponse', supports_download=False)