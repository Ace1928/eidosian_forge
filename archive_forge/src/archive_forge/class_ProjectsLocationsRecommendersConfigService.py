from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class ProjectsLocationsRecommendersConfigService(base_api.BaseApiService):
    """Service class for the projects_locations_recommenders_config resource."""
    _NAME = 'projects_locations_recommenders_config'

    def __init__(self, client):
        super(RecommenderV1alpha2.ProjectsLocationsRecommendersConfigService, self).__init__(client)
        self._upload_configs = {}

    def Commit(self, request, global_params=None):
        """Commits a Recommender Config change.

      Args:
        request: (GoogleCloudRecommenderV1alpha2RecommenderConfig) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2RecommenderConfig) The response message.
      """
        config = self.GetMethodConfig('Commit')
        return self._RunMethod(config, request, global_params=global_params)
    Commit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/recommenders/{recommendersId}/config:commit', http_method='POST', method_id='recommender.projects.locations.recommenders.config.commit', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:commit', request_field='<request>', request_type_name='GoogleCloudRecommenderV1alpha2RecommenderConfig', response_type_name='GoogleCloudRecommenderV1alpha2RecommenderConfig', supports_download=False)