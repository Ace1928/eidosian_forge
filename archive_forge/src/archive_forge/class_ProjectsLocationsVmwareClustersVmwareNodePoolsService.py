from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsVmwareClustersVmwareNodePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_vmwareClusters_vmwareNodePools resource."""
    _NAME = 'projects_locations_vmwareClusters_vmwareNodePools'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsVmwareClustersVmwareNodePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new VMware node pool in a given project, location and VMWare cluster.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools', http_method='POST', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly', 'vmwareNodePoolId'], relative_path='v1/{+parent}/vmwareNodePools', request_field='vmwareNodePool', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single VMware node pool.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools/{vmwareNodePoolsId}', http_method='DELETE', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Enroll(self, request, global_params=None):
        """Enrolls a VMware node pool to Anthos On-Prem API.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsEnrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enroll')
        return self._RunMethod(config, request, global_params=global_params)
    Enroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools:enroll', http_method='POST', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.enroll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/vmwareNodePools:enroll', request_field='enrollVmwareNodePoolRequest', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsEnrollRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single VMware node pool.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VmwareNodePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools/{vmwareNodePoolsId}', http_method='GET', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsGetRequest', response_type_name='VmwareNodePool', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools/{vmwareNodePoolsId}:getIamPolicy', http_method='GET', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists VMware node pools in a given project, location and VMWare cluster.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVmwareNodePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools', http_method='GET', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/vmwareNodePools', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsListRequest', response_type_name='ListVmwareNodePoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single VMware node pool.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools/{vmwareNodePoolsId}', http_method='PATCH', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='vmwareNodePool', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools/{vmwareNodePoolsId}:setIamPolicy', http_method='POST', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools/{vmwareNodePoolsId}:testIamPermissions', http_method='POST', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Unenroll(self, request, global_params=None):
        """Unenrolls a VMware node pool to Anthos On-Prem API.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsUnenrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unenroll')
        return self._RunMethod(config, request, global_params=global_params)
    Unenroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareClusters/{vmwareClustersId}/vmwareNodePools/{vmwareNodePoolsId}:unenroll', http_method='DELETE', method_id='gkeonprem.projects.locations.vmwareClusters.vmwareNodePools.unenroll', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v1/{+name}:unenroll', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsUnenrollRequest', response_type_name='Operation', supports_download=False)