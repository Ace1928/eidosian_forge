from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadIdentityPools_namespaces_managedIdentities_workloadSources resource."""
    _NAME = 'projects_locations_workloadIdentityPools_namespaces_managedIdentities_workloadSources'

    def __init__(self, client):
        super(IamV1.ProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkloadSource.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}/workloadSources', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.workloadSources.create', ordered_params=['parent'], path_params=['parent'], query_params=['workloadSourceId'], relative_path='v1/{+parent}/workloadSources', request_field='workloadSource', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkloadSource.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}/workloadSources/{workloadSourcesId}', http_method='DELETE', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.workloadSources.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkloadSource.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkloadSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}/workloadSources/{workloadSourcesId}', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.workloadSources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesGetRequest', response_type_name='WorkloadSource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all WorkloadSources for a WorkloadIdentityPoolNamespace or WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}/workloadSources', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.workloadSources.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/workloadSources', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesListRequest', response_type_name='ListWorkloadSourcesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkloadSource.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}/workloadSources/{workloadSourcesId}', http_method='PATCH', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.workloadSources.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='workloadSource', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesPatchRequest', response_type_name='Operation', supports_download=False)