from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsRegionsJobsService(base_api.BaseApiService):
    """Service class for the projects_regions_jobs resource."""
    _NAME = 'projects_regions_jobs'

    def __init__(self, client):
        super(DataprocV1.ProjectsRegionsJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts a job cancellation request. To access the job resource after cancellation, call regions/{region}/jobs.list (https://cloud.google.com/dataproc/docs/reference/rest/v1/projects.regions.jobs/list) or regions/{region}/jobs.get (https://cloud.google.com/dataproc/docs/reference/rest/v1/projects.regions.jobs/get).

      Args:
        request: (DataprocProjectsRegionsJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.jobs.cancel', ordered_params=['projectId', 'region', 'jobId'], path_params=['jobId', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/jobs/{jobId}:cancel', request_field='cancelJobRequest', request_type_name='DataprocProjectsRegionsJobsCancelRequest', response_type_name='Job', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the job from the project. If the job is active, the delete fails, and the response returns FAILED_PRECONDITION.

      Args:
        request: (DataprocProjectsRegionsJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='dataproc.projects.regions.jobs.delete', ordered_params=['projectId', 'region', 'jobId'], path_params=['jobId', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/jobs/{jobId}', request_field='', request_type_name='DataprocProjectsRegionsJobsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the resource representation for a job in a project.

      Args:
        request: (DataprocProjectsRegionsJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataproc.projects.regions.jobs.get', ordered_params=['projectId', 'region', 'jobId'], path_params=['jobId', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/jobs/{jobId}', request_field='', request_type_name='DataprocProjectsRegionsJobsGetRequest', response_type_name='Job', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataprocProjectsRegionsJobsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/jobs/{jobsId}:getIamPolicy', http_method='POST', method_id='dataproc.projects.regions.jobs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DataprocProjectsRegionsJobsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetJobAsTemplate(self, request, global_params=None):
        """Exports the resource representation for a job in a project as a template that can be used as a SubmitJobRequest.

      Args:
        request: (DataprocProjectsRegionsJobsGetJobAsTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('GetJobAsTemplate')
        return self._RunMethod(config, request, global_params=global_params)
    GetJobAsTemplate.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataproc.projects.regions.jobs.getJobAsTemplate', ordered_params=['projectId', 'region', 'jobId'], path_params=['jobId', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/jobs/{jobId}:getJobAsTemplate', request_field='', request_type_name='DataprocProjectsRegionsJobsGetJobAsTemplateRequest', response_type_name='Job', supports_download=False)

    def List(self, request, global_params=None):
        """Lists regions/{region}/jobs in a project.

      Args:
        request: (DataprocProjectsRegionsJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataproc.projects.regions.jobs.list', ordered_params=['projectId', 'region'], path_params=['projectId', 'region'], query_params=['clusterName', 'filter', 'jobStateMatcher', 'pageSize', 'pageToken'], relative_path='v1/projects/{projectId}/regions/{region}/jobs', request_field='', request_type_name='DataprocProjectsRegionsJobsListRequest', response_type_name='ListJobsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a job in a project.

      Args:
        request: (DataprocProjectsRegionsJobsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='dataproc.projects.regions.jobs.patch', ordered_params=['projectId', 'region', 'jobId'], path_params=['jobId', 'projectId', 'region'], query_params=['updateMask'], relative_path='v1/projects/{projectId}/regions/{region}/jobs/{jobId}', request_field='job', request_type_name='DataprocProjectsRegionsJobsPatchRequest', response_type_name='Job', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataprocProjectsRegionsJobsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/jobs/{jobsId}:setIamPolicy', http_method='POST', method_id='dataproc.projects.regions.jobs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DataprocProjectsRegionsJobsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Submit(self, request, global_params=None):
        """Submits a job to a cluster.

      Args:
        request: (DataprocProjectsRegionsJobsSubmitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Submit')
        return self._RunMethod(config, request, global_params=global_params)
    Submit.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.jobs.submit', ordered_params=['projectId', 'region'], path_params=['projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/jobs:submit', request_field='submitJobRequest', request_type_name='DataprocProjectsRegionsJobsSubmitRequest', response_type_name='Job', supports_download=False)

    def SubmitAsOperation(self, request, global_params=None):
        """Submits job to a cluster.

      Args:
        request: (DataprocProjectsRegionsJobsSubmitAsOperationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SubmitAsOperation')
        return self._RunMethod(config, request, global_params=global_params)
    SubmitAsOperation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.jobs.submitAsOperation', ordered_params=['projectId', 'region'], path_params=['projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/jobs:submitAsOperation', request_field='submitJobRequest', request_type_name='DataprocProjectsRegionsJobsSubmitAsOperationRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataprocProjectsRegionsJobsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/jobs/{jobsId}:testIamPermissions', http_method='POST', method_id='dataproc.projects.regions.jobs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DataprocProjectsRegionsJobsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)