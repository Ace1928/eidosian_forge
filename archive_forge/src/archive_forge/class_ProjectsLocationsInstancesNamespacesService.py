from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datafusion.v1beta1 import datafusion_v1beta1_messages as messages
class ProjectsLocationsInstancesNamespacesService(base_api.BaseApiService):
    """Service class for the projects_locations_instances_namespaces resource."""
    _NAME = 'projects_locations_instances_namespaces'

    def __init__(self, client):
        super(DatafusionV1beta1.ProjectsLocationsInstancesNamespacesService, self).__init__(client)
        self._upload_configs = {}

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DatafusionProjectsLocationsInstancesNamespacesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/namespaces/{namespacesId}:getIamPolicy', http_method='GET', method_id='datafusion.projects.locations.instances.namespaces.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta1/{+resource}:getIamPolicy', request_field='', request_type_name='DatafusionProjectsLocationsInstancesNamespacesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """List namespaces in a given instance.

      Args:
        request: (DatafusionProjectsLocationsInstancesNamespacesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNamespacesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/namespaces', http_method='GET', method_id='datafusion.projects.locations.instances.namespaces.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1beta1/{+parent}/namespaces', request_field='', request_type_name='DatafusionProjectsLocationsInstancesNamespacesListRequest', response_type_name='ListNamespacesResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (DatafusionProjectsLocationsInstancesNamespacesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/namespaces/{namespacesId}:setIamPolicy', http_method='POST', method_id='datafusion.projects.locations.instances.namespaces.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatafusionProjectsLocationsInstancesNamespacesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DatafusionProjectsLocationsInstancesNamespacesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/namespaces/{namespacesId}:testIamPermissions', http_method='POST', method_id='datafusion.projects.locations.instances.namespaces.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatafusionProjectsLocationsInstancesNamespacesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)