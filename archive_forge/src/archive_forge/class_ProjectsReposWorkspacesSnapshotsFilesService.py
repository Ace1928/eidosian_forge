from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
class ProjectsReposWorkspacesSnapshotsFilesService(base_api.BaseApiService):
    """Service class for the projects_repos_workspaces_snapshots_files resource."""
    _NAME = 'projects_repos_workspaces_snapshots_files'

    def __init__(self, client):
        super(SourceV1.ProjectsReposWorkspacesSnapshotsFilesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Read is given a SourceContext and path, and returns.
file or directory information about that path.

      Args:
        request: (SourceProjectsReposWorkspacesSnapshotsFilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReadResponse) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}/snapshots/{snapshotId}/files/{filesId}', http_method='GET', method_id='source.projects.repos.workspaces.snapshots.files.get', ordered_params=['projectId', 'repoName', 'name', 'snapshotId', 'path'], path_params=['name', 'path', 'projectId', 'repoName', 'snapshotId'], query_params=['pageSize', 'pageToken', 'sourceContext_cloudRepo_aliasContext_kind', 'sourceContext_cloudRepo_aliasContext_name', 'sourceContext_cloudRepo_aliasName', 'sourceContext_cloudRepo_repoId_projectRepoId_projectId', 'sourceContext_cloudRepo_repoId_projectRepoId_repoName', 'sourceContext_cloudRepo_repoId_uid', 'sourceContext_cloudRepo_revisionId', 'sourceContext_cloudWorkspace_workspaceId_repoId_uid', 'sourceContext_gerrit_aliasContext_kind', 'sourceContext_gerrit_aliasContext_name', 'sourceContext_gerrit_aliasName', 'sourceContext_gerrit_gerritProject', 'sourceContext_gerrit_hostUri', 'sourceContext_gerrit_revisionId', 'sourceContext_git_revisionId', 'sourceContext_git_url', 'startPosition'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}/snapshots/{snapshotId}/files/{+path}', request_field='', request_type_name='SourceProjectsReposWorkspacesSnapshotsFilesGetRequest', response_type_name='ReadResponse', supports_download=False)