from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsFeaturestoresEntityTypesFeaturesService(base_api.BaseApiService):
    """Service class for the projects_locations_featurestores_entityTypes_features resource."""
    _NAME = 'projects_locations_featurestores_entityTypes_features'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsFeaturestoresEntityTypesFeaturesService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Creates a batch of Features in a given EntityType.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}/features:batchCreate', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.features.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/features:batchCreate', request_field='googleCloudAiplatformV1BatchCreateFeaturesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesBatchCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Feature in a given EntityType.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}/features', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.features.create', ordered_params=['parent'], path_params=['parent'], query_params=['featureId'], relative_path='v1/{+parent}/features', request_field='googleCloudAiplatformV1Feature', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Feature.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}/features/{featuresId}', http_method='DELETE', method_id='aiplatform.projects.locations.featurestores.entityTypes.features.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Feature.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Feature) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}/features/{featuresId}', http_method='GET', method_id='aiplatform.projects.locations.featurestores.entityTypes.features.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesGetRequest', response_type_name='GoogleCloudAiplatformV1Feature', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Features in a given EntityType.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListFeaturesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}/features', http_method='GET', method_id='aiplatform.projects.locations.featurestores.entityTypes.features.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'latestStatsCount', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/features', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesListRequest', response_type_name='GoogleCloudAiplatformV1ListFeaturesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Feature.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Feature) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}/features/{featuresId}', http_method='PATCH', method_id='aiplatform.projects.locations.featurestores.entityTypes.features.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Feature', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesPatchRequest', response_type_name='GoogleCloudAiplatformV1Feature', supports_download=False)