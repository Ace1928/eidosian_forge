from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
class ProjectsLocationsKeyRingsImportJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_keyRings_importJobs resource."""
    _NAME = 'projects_locations_keyRings_importJobs'

    def __init__(self, client):
        super(CloudkmsV1.ProjectsLocationsKeyRingsImportJobsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new ImportJob within a KeyRing. ImportJob.import_method is required.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsImportJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImportJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/importJobs', http_method='POST', method_id='cloudkms.projects.locations.keyRings.importJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=['importJobId'], relative_path='v1/{+parent}/importJobs', request_field='importJob', request_type_name='CloudkmsProjectsLocationsKeyRingsImportJobsCreateRequest', response_type_name='ImportJob', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns metadata for a given ImportJob.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsImportJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImportJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/importJobs/{importJobsId}', http_method='GET', method_id='cloudkms.projects.locations.keyRings.importJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudkmsProjectsLocationsKeyRingsImportJobsGetRequest', response_type_name='ImportJob', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsImportJobsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/importJobs/{importJobsId}:getIamPolicy', http_method='GET', method_id='cloudkms.projects.locations.keyRings.importJobs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='CloudkmsProjectsLocationsKeyRingsImportJobsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ImportJobs.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsImportJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListImportJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/importJobs', http_method='GET', method_id='cloudkms.projects.locations.keyRings.importJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/importJobs', request_field='', request_type_name='CloudkmsProjectsLocationsKeyRingsImportJobsListRequest', response_type_name='ListImportJobsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsImportJobsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/importJobs/{importJobsId}:setIamPolicy', http_method='POST', method_id='cloudkms.projects.locations.keyRings.importJobs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CloudkmsProjectsLocationsKeyRingsImportJobsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsImportJobsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/importJobs/{importJobsId}:testIamPermissions', http_method='POST', method_id='cloudkms.projects.locations.keyRings.importJobs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CloudkmsProjectsLocationsKeyRingsImportJobsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)