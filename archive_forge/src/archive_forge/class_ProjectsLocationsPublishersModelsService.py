from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsPublishersModelsService(base_api.BaseApiService):
    """Service class for the projects_locations_publishers_models resource."""
    _NAME = 'projects_locations_publishers_models'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsPublishersModelsService, self).__init__(client)
        self._upload_configs = {}

    def GenerateContent(self, request, global_params=None):
        """Generate content with multimodal inputs.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsGenerateContentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1GenerateContentResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateContent')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateContent.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publishers/{publishersId}/models/{modelsId}:generateContent', http_method='POST', method_id='aiplatform.projects.locations.publishers.models.generateContent', ordered_params=['model'], path_params=['model'], query_params=[], relative_path='v1/{+model}:generateContent', request_field='googleCloudAiplatformV1GenerateContentRequest', request_type_name='AiplatformProjectsLocationsPublishersModelsGenerateContentRequest', response_type_name='GoogleCloudAiplatformV1GenerateContentResponse', supports_download=False)

    def Predict(self, request, global_params=None):
        """Perform an online prediction.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1PredictResponse) The response message.
      """
        config = self.GetMethodConfig('Predict')
        return self._RunMethod(config, request, global_params=global_params)
    Predict.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publishers/{publishersId}/models/{modelsId}:predict', http_method='POST', method_id='aiplatform.projects.locations.publishers.models.predict', ordered_params=['endpoint'], path_params=['endpoint'], query_params=[], relative_path='v1/{+endpoint}:predict', request_field='googleCloudAiplatformV1PredictRequest', request_type_name='AiplatformProjectsLocationsPublishersModelsPredictRequest', response_type_name='GoogleCloudAiplatformV1PredictResponse', supports_download=False)

    def RawPredict(self, request, global_params=None):
        """Perform an online prediction with an arbitrary HTTP payload. The response includes the following HTTP headers: * `X-Vertex-AI-Endpoint-Id`: ID of the Endpoint that served this prediction. * `X-Vertex-AI-Deployed-Model-Id`: ID of the Endpoint's DeployedModel that served this prediction.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsRawPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('RawPredict')
        return self._RunMethod(config, request, global_params=global_params)
    RawPredict.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publishers/{publishersId}/models/{modelsId}:rawPredict', http_method='POST', method_id='aiplatform.projects.locations.publishers.models.rawPredict', ordered_params=['endpoint'], path_params=['endpoint'], query_params=[], relative_path='v1/{+endpoint}:rawPredict', request_field='googleCloudAiplatformV1RawPredictRequest', request_type_name='AiplatformProjectsLocationsPublishersModelsRawPredictRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def ServerStreamingPredict(self, request, global_params=None):
        """Perform a server-side streaming online prediction request for Vertex LLM streaming.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsServerStreamingPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1StreamingPredictResponse) The response message.
      """
        config = self.GetMethodConfig('ServerStreamingPredict')
        return self._RunMethod(config, request, global_params=global_params)
    ServerStreamingPredict.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publishers/{publishersId}/models/{modelsId}:serverStreamingPredict', http_method='POST', method_id='aiplatform.projects.locations.publishers.models.serverStreamingPredict', ordered_params=['endpoint'], path_params=['endpoint'], query_params=[], relative_path='v1/{+endpoint}:serverStreamingPredict', request_field='googleCloudAiplatformV1StreamingPredictRequest', request_type_name='AiplatformProjectsLocationsPublishersModelsServerStreamingPredictRequest', response_type_name='GoogleCloudAiplatformV1StreamingPredictResponse', supports_download=False)

    def StreamGenerateContent(self, request, global_params=None):
        """Generate content with multimodal inputs with streaming support.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsStreamGenerateContentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1GenerateContentResponse) The response message.
      """
        config = self.GetMethodConfig('StreamGenerateContent')
        return self._RunMethod(config, request, global_params=global_params)
    StreamGenerateContent.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publishers/{publishersId}/models/{modelsId}:streamGenerateContent', http_method='POST', method_id='aiplatform.projects.locations.publishers.models.streamGenerateContent', ordered_params=['model'], path_params=['model'], query_params=[], relative_path='v1/{+model}:streamGenerateContent', request_field='googleCloudAiplatformV1GenerateContentRequest', request_type_name='AiplatformProjectsLocationsPublishersModelsStreamGenerateContentRequest', response_type_name='GoogleCloudAiplatformV1GenerateContentResponse', supports_download=False)

    def StreamRawPredict(self, request, global_params=None):
        """Perform a streaming online prediction with an arbitrary HTTP payload.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsStreamRawPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('StreamRawPredict')
        return self._RunMethod(config, request, global_params=global_params)
    StreamRawPredict.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publishers/{publishersId}/models/{modelsId}:streamRawPredict', http_method='POST', method_id='aiplatform.projects.locations.publishers.models.streamRawPredict', ordered_params=['endpoint'], path_params=['endpoint'], query_params=[], relative_path='v1/{+endpoint}:streamRawPredict', request_field='googleCloudAiplatformV1StreamRawPredictRequest', request_type_name='AiplatformProjectsLocationsPublishersModelsStreamRawPredictRequest', response_type_name='GoogleApiHttpBody', supports_download=False)