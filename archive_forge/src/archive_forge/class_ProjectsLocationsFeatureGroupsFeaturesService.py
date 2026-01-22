from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsFeatureGroupsFeaturesService(base_api.BaseApiService):
    """Service class for the projects_locations_featureGroups_features resource."""
    _NAME = 'projects_locations_featureGroups_features'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsFeatureGroupsFeaturesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Feature in a given FeatureGroup.

      Args:
        request: (AiplatformProjectsLocationsFeatureGroupsFeaturesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureGroups/{featureGroupsId}/features', http_method='POST', method_id='aiplatform.projects.locations.featureGroups.features.create', ordered_params=['parent'], path_params=['parent'], query_params=['featureId'], relative_path='v1/{+parent}/features', request_field='googleCloudAiplatformV1Feature', request_type_name='AiplatformProjectsLocationsFeatureGroupsFeaturesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Feature.

      Args:
        request: (AiplatformProjectsLocationsFeatureGroupsFeaturesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureGroups/{featureGroupsId}/features/{featuresId}', http_method='DELETE', method_id='aiplatform.projects.locations.featureGroups.features.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeatureGroupsFeaturesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Feature.

      Args:
        request: (AiplatformProjectsLocationsFeatureGroupsFeaturesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Feature) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureGroups/{featureGroupsId}/features/{featuresId}', http_method='GET', method_id='aiplatform.projects.locations.featureGroups.features.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeatureGroupsFeaturesGetRequest', response_type_name='GoogleCloudAiplatformV1Feature', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Features in a given FeatureGroup.

      Args:
        request: (AiplatformProjectsLocationsFeatureGroupsFeaturesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListFeaturesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureGroups/{featureGroupsId}/features', http_method='GET', method_id='aiplatform.projects.locations.featureGroups.features.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'latestStatsCount', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/features', request_field='', request_type_name='AiplatformProjectsLocationsFeatureGroupsFeaturesListRequest', response_type_name='GoogleCloudAiplatformV1ListFeaturesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Feature.

      Args:
        request: (AiplatformProjectsLocationsFeatureGroupsFeaturesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureGroups/{featureGroupsId}/features/{featuresId}', http_method='PATCH', method_id='aiplatform.projects.locations.featureGroups.features.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Feature', request_type_name='AiplatformProjectsLocationsFeatureGroupsFeaturesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)