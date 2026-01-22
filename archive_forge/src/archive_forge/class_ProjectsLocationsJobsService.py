from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
class ProjectsLocationsJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_jobs resource."""
    _NAME = 'projects_locations_jobs'

    def __init__(self, client):
        super(RunV2.ProjectsLocationsJobsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Job.

      Args:
        request: (RunProjectsLocationsJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs', http_method='POST', method_id='run.projects.locations.jobs.create', ordered_params=['parent'], path_params=['parent'], query_params=['jobId', 'validateOnly'], relative_path='v2/{+parent}/jobs', request_field='googleCloudRunV2Job', request_type_name='RunProjectsLocationsJobsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Job.

      Args:
        request: (RunProjectsLocationsJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}', http_method='DELETE', method_id='run.projects.locations.jobs.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a Job.

      Args:
        request: (RunProjectsLocationsJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2Job) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}', http_method='GET', method_id='run.projects.locations.jobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsJobsGetRequest', response_type_name='GoogleCloudRunV2Job', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM Access Control policy currently in effect for the given Job. This result does not include any inherited policies.

      Args:
        request: (RunProjectsLocationsJobsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}:getIamPolicy', http_method='GET', method_id='run.projects.locations.jobs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v2/{+resource}:getIamPolicy', request_field='', request_type_name='RunProjectsLocationsJobsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Jobs.

      Args:
        request: (RunProjectsLocationsJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ListJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs', http_method='GET', method_id='run.projects.locations.jobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/jobs', request_field='', request_type_name='RunProjectsLocationsJobsListRequest', response_type_name='GoogleCloudRunV2ListJobsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Job.

      Args:
        request: (RunProjectsLocationsJobsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}', http_method='PATCH', method_id='run.projects.locations.jobs.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'validateOnly'], relative_path='v2/{+name}', request_field='googleCloudRunV2Job', request_type_name='RunProjectsLocationsJobsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Run(self, request, global_params=None):
        """Triggers creation of a new Execution of this Job.

      Args:
        request: (RunProjectsLocationsJobsRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}:run', http_method='POST', method_id='run.projects.locations.jobs.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:run', request_field='googleCloudRunV2RunJobRequest', request_type_name='RunProjectsLocationsJobsRunRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM Access control policy for the specified Job. Overwrites any existing policy.

      Args:
        request: (RunProjectsLocationsJobsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}:setIamPolicy', http_method='POST', method_id='run.projects.locations.jobs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='RunProjectsLocationsJobsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified Project. There are no permissions required for making this API call.

      Args:
        request: (RunProjectsLocationsJobsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}:testIamPermissions', http_method='POST', method_id='run.projects.locations.jobs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='RunProjectsLocationsJobsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)