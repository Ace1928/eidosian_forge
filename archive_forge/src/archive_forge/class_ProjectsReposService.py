from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sourcerepo.v1 import sourcerepo_v1_messages as messages
class ProjectsReposService(base_api.BaseApiService):
    """Service class for the projects_repos resource."""
    _NAME = 'projects_repos'

    def __init__(self, client):
        super(SourcerepoV1.ProjectsReposService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a repo in the given project with the given name. If the named repository already exists, `CreateRepo` returns `ALREADY_EXISTS`.

      Args:
        request: (SourcerepoProjectsReposCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Repo) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos', http_method='POST', method_id='sourcerepo.projects.repos.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/repos', request_field='repo', request_type_name='SourcerepoProjectsReposCreateRequest', response_type_name='Repo', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a repo.

      Args:
        request: (SourcerepoProjectsReposDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos/{reposId}', http_method='DELETE', method_id='sourcerepo.projects.repos.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SourcerepoProjectsReposDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about a repo.

      Args:
        request: (SourcerepoProjectsReposGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Repo) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos/{reposId}', http_method='GET', method_id='sourcerepo.projects.repos.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SourcerepoProjectsReposGetRequest', response_type_name='Repo', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (SourcerepoProjectsReposGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos/{reposId}:getIamPolicy', http_method='GET', method_id='sourcerepo.projects.repos.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='SourcerepoProjectsReposGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns all repos belonging to a project. The sizes of the repos are not set by ListRepos. To get the size of a repo, use GetRepo.

      Args:
        request: (SourcerepoProjectsReposListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReposResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos', http_method='GET', method_id='sourcerepo.projects.repos.list', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+name}/repos', request_field='', request_type_name='SourcerepoProjectsReposListRequest', response_type_name='ListReposResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates information about a repo.

      Args:
        request: (SourcerepoProjectsReposPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Repo) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos/{reposId}', http_method='PATCH', method_id='sourcerepo.projects.repos.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='updateRepoRequest', request_type_name='SourcerepoProjectsReposPatchRequest', response_type_name='Repo', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy on the specified resource. Replaces any existing policy.

      Args:
        request: (SourcerepoProjectsReposSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos/{reposId}:setIamPolicy', http_method='POST', method_id='sourcerepo.projects.repos.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='SourcerepoProjectsReposSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Sync(self, request, global_params=None):
        """Synchronize a connected repo. The response contains SyncRepoMetadata in the metadata field.

      Args:
        request: (SourcerepoProjectsReposSyncRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Sync')
        return self._RunMethod(config, request, global_params=global_params)
    Sync.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos/{reposId}:sync', http_method='POST', method_id='sourcerepo.projects.repos.sync', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:sync', request_field='syncRepoRequest', request_type_name='SourcerepoProjectsReposSyncRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.

      Args:
        request: (SourcerepoProjectsReposTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/repos/{reposId}:testIamPermissions', http_method='POST', method_id='sourcerepo.projects.repos.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='SourcerepoProjectsReposTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)