from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class RecommendersService(base_api.BaseApiService):
    """Service class for the recommenders resource."""
    _NAME = 'recommenders'

    def __init__(self, client):
        super(RecommenderV1alpha2.RecommendersService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all available Recommenders. No IAM permissions are required.

      Args:
        request: (RecommenderRecommendersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2ListRecommendersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='recommender.recommenders.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken'], relative_path='v1alpha2/recommenders', request_field='', request_type_name='RecommenderRecommendersListRequest', response_type_name='GoogleCloudRecommenderV1alpha2ListRecommendersResponse', supports_download=False)