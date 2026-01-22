from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsLocationsWorkloadIdentityPoolsNamespacesService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadIdentityPools_namespaces resource."""
    _NAME = 'projects_locations_workloadIdentityPools_namespaces'

    def __init__(self, client):
        super(IamV1.ProjectsLocationsWorkloadIdentityPoolsNamespacesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkloadIdentityPoolNamespace in a WorkloadIdentityPool.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.create', ordered_params=['parent'], path_params=['parent'], query_params=['workloadIdentityPoolNamespaceId'], relative_path='v1/{+parent}/namespaces', request_field='workloadIdentityPoolNamespace', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkloadIdentityPoolNamespace. You can undelete a namespace for 30 days. After 30 days, deletion is permanent.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}', http_method='DELETE', method_id='iam.projects.locations.workloadIdentityPools.namespaces.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkloadIdentityPoolNamespace.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkloadIdentityPoolNamespace) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.namespaces.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesGetRequest', response_type_name='WorkloadIdentityPoolNamespace', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets IAM policies for one of WorkloadIdentityPool WorkloadIdentityPoolNamespace WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}:getIamPolicy', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkloadIdentityPoolNamespaces in a workload identity pool. If `show_deleted` is set to `true`, then deleted namespaces are also listed.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadIdentityPoolNamespacesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.namespaces.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/namespaces', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesListRequest', response_type_name='ListWorkloadIdentityPoolNamespacesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkloadIdentityPoolNamespace in a WorkloadIdentityPool.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}', http_method='PATCH', method_id='iam.projects.locations.workloadIdentityPools.namespaces.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='workloadIdentityPoolNamespace', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets IAM policies on one of WorkloadIdentityPool WorkloadIdentityPoolNamespace WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}:setIamPolicy', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the caller's permissions on one of WorkloadIdentityPool WorkloadIdentityPoolNamespace WorkloadIdentityPoolManagedIdentity.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}:testIamPermissions', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkloadIdentityPoolNamespace, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsNamespacesUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/namespaces/{namespacesId}:undelete', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.namespaces.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkloadIdentityPoolNamespaceRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsNamespacesUndeleteRequest', response_type_name='Operation', supports_download=False)