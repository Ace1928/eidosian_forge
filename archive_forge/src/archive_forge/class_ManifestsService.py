from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.deploymentmanager.v2 import deploymentmanager_v2_messages as messages
class ManifestsService(base_api.BaseApiService):
    """Service class for the manifests resource."""
    _NAME = 'manifests'

    def __init__(self, client):
        super(DeploymentmanagerV2.ManifestsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets information about a specific manifest.

      Args:
        request: (DeploymentmanagerManifestsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Manifest) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.manifests.get', ordered_params=['project', 'deployment', 'manifest'], path_params=['deployment', 'manifest', 'project'], query_params=[], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}/manifests/{manifest}', request_field='', request_type_name='DeploymentmanagerManifestsGetRequest', response_type_name='Manifest', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all manifests for a given deployment.

      Args:
        request: (DeploymentmanagerManifestsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManifestsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.manifests.list', ordered_params=['project', 'deployment'], path_params=['deployment', 'project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken'], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}/manifests', request_field='', request_type_name='DeploymentmanagerManifestsListRequest', response_type_name='ManifestsListResponse', supports_download=False)