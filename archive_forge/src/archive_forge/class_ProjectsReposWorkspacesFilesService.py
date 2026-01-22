from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
class ProjectsReposWorkspacesFilesService(base_api.BaseApiService):
    """Service class for the projects_repos_workspaces_files resource."""
    _NAME = 'projects_repos_workspaces_files'

    def __init__(self, client):
        super(SourceV1.ProjectsReposWorkspacesFilesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Read is given a SourceContext and path, and returns.
file or directory information about that path.

      Args:
        request: (SourceProjectsReposWorkspacesFilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReadResponse) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}/files/{filesId}', http_method='GET', method_id='source.projects.repos.workspaces.files.get', ordered_params=['projectId', 'repoName', 'name', 'path'], path_params=['name', 'path', 'projectId', 'repoName'], query_params=['pageSize', 'pageToken', 'sourceContext_cloudRepo_aliasContext_kind', 'sourceContext_cloudRepo_aliasContext_name', 'sourceContext_cloudRepo_aliasName', 'sourceContext_cloudRepo_repoId_projectRepoId_projectId', 'sourceContext_cloudRepo_repoId_projectRepoId_repoName', 'sourceContext_cloudRepo_repoId_uid', 'sourceContext_cloudRepo_revisionId', 'sourceContext_cloudWorkspace_snapshotId', 'sourceContext_cloudWorkspace_workspaceId_repoId_uid', 'sourceContext_gerrit_aliasContext_kind', 'sourceContext_gerrit_aliasContext_name', 'sourceContext_gerrit_aliasName', 'sourceContext_gerrit_gerritProject', 'sourceContext_gerrit_hostUri', 'sourceContext_gerrit_revisionId', 'sourceContext_git_revisionId', 'sourceContext_git_url', 'startPosition'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}/files/{+path}', request_field='', request_type_name='SourceProjectsReposWorkspacesFilesGetRequest', response_type_name='ReadResponse', supports_download=False)