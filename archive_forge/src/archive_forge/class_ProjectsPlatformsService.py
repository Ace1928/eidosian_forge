from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1 import binaryauthorization_v1_messages as messages
class ProjectsPlatformsService(base_api.BaseApiService):
    """Service class for the projects_platforms resource."""
    _NAME = 'projects_platforms'

    def __init__(self, client):
        super(BinaryauthorizationV1.ProjectsPlatformsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all platforms supported by the platform policy.

      Args:
        request: (BinaryauthorizationProjectsPlatformsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPlatformsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/platforms', http_method='GET', method_id='binaryauthorization.projects.platforms.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/platforms', request_field='', request_type_name='BinaryauthorizationProjectsPlatformsListRequest', response_type_name='ListPlatformsResponse', supports_download=False)