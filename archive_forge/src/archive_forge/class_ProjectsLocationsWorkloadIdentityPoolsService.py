from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1beta import iam_v1beta_messages as messages
class ProjectsLocationsWorkloadIdentityPoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadIdentityPools resource."""
    _NAME = 'projects_locations_workloadIdentityPools'

    def __init__(self, client):
        super(IamV1beta.ProjectsLocationsWorkloadIdentityPoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkloadIdentityPool. You cannot reuse the name of a deleted pool until 30 days after deletion.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.create', ordered_params=['parent'], path_params=['parent'], query_params=['workloadIdentityPoolId'], relative_path='v1beta/{+parent}/workloadIdentityPools', request_field='googleIamV1betaWorkloadIdentityPool', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkloadIdentityPool. You cannot use a deleted pool to exchange external credentials for Google Cloud credentials. However, deletion does not revoke credentials that have already been issued. Credentials issued for a deleted pool do not grant access to resources. If the pool is undeleted, and the credentials are not expired, they grant access again. You can undelete a pool for 30 days. After 30 days, deletion is permanent. You cannot update deleted pools. However, you can view and list them.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}', http_method='DELETE', method_id='iam.projects.locations.workloadIdentityPools.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkloadIdentityPool.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1betaWorkloadIdentityPool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsGetRequest', response_type_name='GoogleIamV1betaWorkloadIdentityPool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkloadIdentityPools in a project. If `show_deleted` is set to `true`, then deleted pools are also listed.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1betaListWorkloadIdentityPoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1beta/{+parent}/workloadIdentityPools', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsListRequest', response_type_name='GoogleIamV1betaListWorkloadIdentityPoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkloadIdentityPool.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}', http_method='PATCH', method_id='iam.projects.locations.workloadIdentityPools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='googleIamV1betaWorkloadIdentityPool', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkloadIdentityPool, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}:undelete', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:undelete', request_field='googleIamV1betaUndeleteWorkloadIdentityPoolRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsUndeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)