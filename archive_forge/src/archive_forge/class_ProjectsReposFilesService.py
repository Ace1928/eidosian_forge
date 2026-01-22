from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
class ProjectsReposFilesService(base_api.BaseApiService):
    """Service class for the projects_repos_files resource."""
    _NAME = 'projects_repos_files'

    def __init__(self, client):
        super(SourceV1.ProjectsReposFilesService, self).__init__(client)
        self._upload_configs = {}

    def ReadFromWorkspaceOrAlias(self, request, global_params=None):
        """ReadFromWorkspaceOrAlias performs a Read using either the most recent.
snapshot of the given workspace, if the workspace exists, or the
revision referred to by the given alias if the workspace does not exist.

      Args:
        request: (SourceProjectsReposFilesReadFromWorkspaceOrAliasRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReadResponse) The response message.
      """
        config = self.GetMethodConfig('ReadFromWorkspaceOrAlias')
        return self._RunMethod(config, request, global_params=global_params)
    ReadFromWorkspaceOrAlias.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectId}/repos/{repoName}/files/{filesId}:readFromWorkspaceOrAlias', http_method='GET', method_id='source.projects.repos.files.readFromWorkspaceOrAlias', ordered_params=['projectId', 'repoName', 'path'], path_params=['path', 'projectId', 'repoName'], query_params=['alias', 'pageSize', 'pageToken', 'repoId_uid', 'startPosition', 'workspaceName'], relative_path='v1/projects/{projectId}/repos/{repoName}/files/{+path}:readFromWorkspaceOrAlias', request_field='', request_type_name='SourceProjectsReposFilesReadFromWorkspaceOrAliasRequest', response_type_name='ReadResponse', supports_download=False)