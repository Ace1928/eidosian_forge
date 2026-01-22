from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1beta import iam_v1beta_messages as messages
class ProjectsLocationsWorkloadIdentityPoolsProvidersOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadIdentityPools_providers_operations resource."""
    _NAME = 'projects_locations_workloadIdentityPools_providers_operations'

    def __init__(self, client):
        super(IamV1beta.ProjectsLocationsWorkloadIdentityPoolsProvidersOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}/operations/{operationsId}', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.providers.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)