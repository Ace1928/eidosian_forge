from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicedirectory.v1beta1 import servicedirectory_v1beta1_messages as messages
class ProjectsLocationsNamespacesWorkloadsService(base_api.BaseApiService):
    """Service class for the projects_locations_namespaces_workloads resource."""
    _NAME = 'projects_locations_namespaces_workloads'

    def __init__(self, client):
        super(ServicedirectoryV1beta1.ProjectsLocationsNamespacesWorkloadsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a workload, and returns the new workload.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workload) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.workloads.create', ordered_params=['parent'], path_params=['parent'], query_params=['workloadId'], relative_path='v1beta1/{+parent}/workloads', request_field='workload', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsCreateRequest', response_type_name='Workload', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a workload.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads/{workloadsId}', http_method='DELETE', method_id='servicedirectory.projects.locations.namespaces.workloads.delete', ordered_params=['name'], path_params=['name'], query_params=['managerType'], relative_path='v1beta1/{+name}', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a service workload.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workload) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads/{workloadsId}', http_method='GET', method_id='servicedirectory.projects.locations.namespaces.workloads.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsGetRequest', response_type_name='Workload', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM Policy for a resource.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads/{workloadsId}:getIamPolicy', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.workloads.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all service workloads.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads', http_method='GET', method_id='servicedirectory.projects.locations.namespaces.workloads.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/workloads', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsListRequest', response_type_name='ListWorkloadsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a workload.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workload) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads/{workloadsId}', http_method='PATCH', method_id='servicedirectory.projects.locations.namespaces.workloads.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='workload', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsPatchRequest', response_type_name='Workload', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM Policy for a resource.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads/{workloadsId}:setIamPolicy', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.workloads.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests IAM permissions for a resource (namespace, service or service workload only).

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesWorkloadsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/workloads/{workloadsId}:testIamPermissions', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.workloads.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ServicedirectoryProjectsLocationsNamespacesWorkloadsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)