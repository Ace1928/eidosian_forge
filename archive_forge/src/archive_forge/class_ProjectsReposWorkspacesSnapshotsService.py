from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
class ProjectsReposWorkspacesSnapshotsService(base_api.BaseApiService):
    """Service class for the projects_repos_workspaces_snapshots resource."""
    _NAME = 'projects_repos_workspaces_snapshots'

    def __init__(self, client):
        super(SourceV1.ProjectsReposWorkspacesSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a workspace snapshot.

      Args:
        request: (SourceProjectsReposWorkspacesSnapshotsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snapshot) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.workspaces.snapshots.get', ordered_params=['projectId', 'repoName', 'name', 'snapshotId'], path_params=['name', 'projectId', 'repoName', 'snapshotId'], query_params=['workspaceId_repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}/snapshots/{snapshotId}', request_field='', request_type_name='SourceProjectsReposWorkspacesSnapshotsGetRequest', response_type_name='Snapshot', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the snapshots made to a workspace, sorted from most recent to.
least recent.

      Args:
        request: (SourceProjectsReposWorkspacesSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSnapshotsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.workspaces.snapshots.list', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=['pageSize', 'pageToken', 'workspaceId_repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}/snapshots', request_field='', request_type_name='SourceProjectsReposWorkspacesSnapshotsListRequest', response_type_name='ListSnapshotsResponse', supports_download=False)

    def ListFiles(self, request, global_params=None):
        """ListFiles returns a list of all files in a SourceContext. The.
information about each file includes its path and its hash.
The result is ordered by path. Pagination is supported.

      Args:
        request: (SourceProjectsReposWorkspacesSnapshotsListFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFilesResponse) The response message.
      """
        config = self.GetMethodConfig('ListFiles')
        return self._RunMethod(config, request, global_params=global_params)
    ListFiles.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.workspaces.snapshots.listFiles', ordered_params=['projectId', 'repoName', 'name', 'snapshotId'], path_params=['name', 'projectId', 'repoName', 'snapshotId'], query_params=['pageSize', 'pageToken', 'sourceContext_cloudRepo_aliasContext_kind', 'sourceContext_cloudRepo_aliasContext_name', 'sourceContext_cloudRepo_aliasName', 'sourceContext_cloudRepo_repoId_projectRepoId_projectId', 'sourceContext_cloudRepo_repoId_projectRepoId_repoName', 'sourceContext_cloudRepo_repoId_uid', 'sourceContext_cloudRepo_revisionId', 'sourceContext_cloudWorkspace_workspaceId_repoId_uid', 'sourceContext_gerrit_aliasContext_kind', 'sourceContext_gerrit_aliasContext_name', 'sourceContext_gerrit_aliasName', 'sourceContext_gerrit_gerritProject', 'sourceContext_gerrit_hostUri', 'sourceContext_gerrit_revisionId', 'sourceContext_git_revisionId', 'sourceContext_git_url'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}/snapshots/{snapshotId}:listFiles', request_field='', request_type_name='SourceProjectsReposWorkspacesSnapshotsListFilesRequest', response_type_name='ListFilesResponse', supports_download=False)