from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsStorageQuotasService(base_api.BaseApiService):
    """Service class for the projects_locations_storageQuotas resource."""
    _NAME = 'projects_locations_storageQuotas'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsStorageQuotasService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List Storage provisioning quotas.

      Args:
        request: (BaremetalsolutionProjectsLocationsStorageQuotasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStorageQuotasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/storageQuotas', http_method='GET', method_id='baremetalsolution.projects.locations.storageQuotas.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/storageQuotas', request_field='', request_type_name='BaremetalsolutionProjectsLocationsStorageQuotasListRequest', response_type_name='ListStorageQuotasResponse', supports_download=False)