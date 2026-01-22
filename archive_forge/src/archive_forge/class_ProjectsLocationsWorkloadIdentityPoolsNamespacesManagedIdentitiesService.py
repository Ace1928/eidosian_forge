from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadIdentityPools_namespaces_managedIdentities resource."""
    _NAME = 'projects_locations_workloadIdentityPools_namespaces_managedIdentities'

    def __init__(self, client):
        super(IamV1.ProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkloadIdentityPoolManagedIdentity in a WorkloadIdentityPoolNamespace.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.create', ordered_params=['parent'], path_params=['parent'], query_params=['workloadIdentityPoolManagedIdentityId'], relative_path='v1/{+parent}/managedIdentities', request_field='workloadIdentityPoolManagedIdentity', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkloadIdentityPoolManagedIdentity. You can undelete a managed identity for 30 days. After 30 days, deletion is permanent.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}', http_method='DELETE', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkloadIdentityPoolManagedIdentity) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesGetRequest', response_type_name='WorkloadIdentityPoolManagedIdentity', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets IAM policies for one of WorkloadIdentityPool WorkloadIdentityPoolNamespace WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}:getIamPolicy', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkloadIdentityPoolManagedIdentitys in a namespace. If `show_deleted` is set to `true`, then deleted managed identites are also listed.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadIdentityPoolManagedIdentitiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/managedIdentities', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesListRequest', response_type_name='ListWorkloadIdentityPoolManagedIdentitiesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkloadIdentityPoolManagedIdentity in a WorkloadIdentityPoolNamespace.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}', http_method='PATCH', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='workloadIdentityPoolManagedIdentity', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets IAM policies on one of WorkloadIdentityPool WorkloadIdentityPoolNamespace WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}:setIamPolicy', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the caller's permissions on one of WorkloadIdentityPool WorkloadIdentityPoolNamespace WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}:testIamPermissions', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkloadIdentityPoolManagedIdentity, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}/managedIdentities/{managedIdentitiesId}:undelete', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkloadIdentityPoolManagedIdentityRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesUndeleteRequest', response_type_name='Operation', supports_download=False)