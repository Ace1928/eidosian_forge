from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class ProjectsLocationsRecommendersService(base_api.BaseApiService):
    """Service class for the projects_locations_recommenders resource."""
    _NAME = 'projects_locations_recommenders'

    def __init__(self, client):
        super(RecommenderV1alpha2.ProjectsLocationsRecommendersService, self).__init__(client)
        self._upload_configs = {}

    def GetConfig(self, request, global_params=None):
        """Gets the requested Recommender Config. There is only one instance of the config for each Recommender.

      Args:
        request: (RecommenderProjectsLocationsRecommendersGetConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2RecommenderConfig) The response message.
      """
        config = self.GetMethodConfig('GetConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/recommenders/{recommendersId}/config', http_method='GET', method_id='recommender.projects.locations.recommenders.getConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='RecommenderProjectsLocationsRecommendersGetConfigRequest', response_type_name='GoogleCloudRecommenderV1alpha2RecommenderConfig', supports_download=False)