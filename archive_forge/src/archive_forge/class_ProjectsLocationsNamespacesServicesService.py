from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicedirectory.v1 import servicedirectory_v1_messages as messages
class ProjectsLocationsNamespacesServicesService(base_api.BaseApiService):
    """Service class for the projects_locations_namespaces_services resource."""
    _NAME = 'projects_locations_namespaces_services'

    def __init__(self, client):
        super(ServicedirectoryV1.ProjectsLocationsNamespacesServicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a service, and returns the new service.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.services.create', ordered_params=['parent'], path_params=['parent'], query_params=['serviceId'], relative_path='v1/{+parent}/services', request_field='service', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesCreateRequest', response_type_name='Service', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a service. This also deletes all endpoints associated with the service.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}', http_method='DELETE', method_id='servicedirectory.projects.locations.namespaces.services.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a service.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}', http_method='GET', method_id='servicedirectory.projects.locations.namespaces.services.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesGetRequest', response_type_name='Service', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM Policy for a resource (namespace or service only).

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}:getIamPolicy', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.services.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all services belonging to a namespace.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services', http_method='GET', method_id='servicedirectory.projects.locations.namespaces.services.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/services', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesListRequest', response_type_name='ListServicesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a service.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}', http_method='PATCH', method_id='servicedirectory.projects.locations.namespaces.services.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='service', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesPatchRequest', response_type_name='Service', supports_download=False)

    def Resolve(self, request, global_params=None):
        """Returns a service and its associated endpoints. Resolving a service is not considered an active developer method.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesResolveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResolveServiceResponse) The response message.
      """
        config = self.GetMethodConfig('Resolve')
        return self._RunMethod(config, request, global_params=global_params)
    Resolve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}:resolve', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.services.resolve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resolve', request_field='resolveServiceRequest', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesResolveRequest', response_type_name='ResolveServiceResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM Policy for a resource (namespace or service only).

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}:setIamPolicy', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.services.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests IAM permissions for a resource (namespace or service only).

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}:testIamPermissions', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.services.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)