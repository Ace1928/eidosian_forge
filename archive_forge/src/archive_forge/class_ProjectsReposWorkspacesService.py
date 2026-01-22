from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
class ProjectsReposWorkspacesService(base_api.BaseApiService):
    """Service class for the projects_repos_workspaces resource."""
    _NAME = 'projects_repos_workspaces'

    def __init__(self, client):
        super(SourceV1.ProjectsReposWorkspacesService, self).__init__(client)
        self._upload_configs = {}

    def CommitWorkspace(self, request, global_params=None):
        """Commits some or all of the modified files in a workspace. This creates a.
new revision in the repo with the workspace's contents. Returns ABORTED if the workspace ID
in the request contains a snapshot ID and it is not the same as the
workspace's current snapshot ID or if the workspace is simultaneously
modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesCommitWorkspaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
        config = self.GetMethodConfig('CommitWorkspace')
        return self._RunMethod(config, request, global_params=global_params)
    CommitWorkspace.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='source.projects.repos.workspaces.commitWorkspace', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=[], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}:commitWorkspace', request_field='commitWorkspaceRequest', request_type_name='SourceProjectsReposWorkspacesCommitWorkspaceRequest', response_type_name='Workspace', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a workspace.

      Args:
        request: (SourceProjectsReposWorkspacesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='source.projects.repos.workspaces.create', ordered_params=['projectId', 'repoName'], path_params=['projectId', 'repoName'], query_params=[], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces', request_field='createWorkspaceRequest', request_type_name='SourceProjectsReposWorkspacesCreateRequest', response_type_name='Workspace', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a workspace. Uncommitted changes are lost. If the workspace does.
not exist, NOT_FOUND is returned. Returns ABORTED when the workspace is
simultaneously modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='source.projects.repos.workspaces.delete', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=['currentSnapshotId', 'workspaceId_repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}', request_field='', request_type_name='SourceProjectsReposWorkspacesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns workspace metadata.

      Args:
        request: (SourceProjectsReposWorkspacesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.workspaces.get', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=['workspaceId_repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}', request_field='', request_type_name='SourceProjectsReposWorkspacesGetRequest', response_type_name='Workspace', supports_download=False)

    def List(self, request, global_params=None):
        """Returns all workspaces belonging to a repo.

      Args:
        request: (SourceProjectsReposWorkspacesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkspacesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.workspaces.list', ordered_params=['projectId', 'repoName'], path_params=['projectId', 'repoName'], query_params=['repoId_uid', 'view'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces', request_field='', request_type_name='SourceProjectsReposWorkspacesListRequest', response_type_name='ListWorkspacesResponse', supports_download=False)

    def ListFiles(self, request, global_params=None):
        """ListFiles returns a list of all files in a SourceContext. The.
information about each file includes its path and its hash.
The result is ordered by path. Pagination is supported.

      Args:
        request: (SourceProjectsReposWorkspacesListFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFilesResponse) The response message.
      """
        config = self.GetMethodConfig('ListFiles')
        return self._RunMethod(config, request, global_params=global_params)
    ListFiles.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.workspaces.listFiles', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=['pageSize', 'pageToken', 'sourceContext_cloudRepo_aliasContext_kind', 'sourceContext_cloudRepo_aliasContext_name', 'sourceContext_cloudRepo_aliasName', 'sourceContext_cloudRepo_repoId_projectRepoId_projectId', 'sourceContext_cloudRepo_repoId_projectRepoId_repoName', 'sourceContext_cloudRepo_repoId_uid', 'sourceContext_cloudRepo_revisionId', 'sourceContext_cloudWorkspace_snapshotId', 'sourceContext_cloudWorkspace_workspaceId_repoId_uid', 'sourceContext_gerrit_aliasContext_kind', 'sourceContext_gerrit_aliasContext_name', 'sourceContext_gerrit_aliasName', 'sourceContext_gerrit_gerritProject', 'sourceContext_gerrit_hostUri', 'sourceContext_gerrit_revisionId', 'sourceContext_git_revisionId', 'sourceContext_git_url'], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}:listFiles', request_field='', request_type_name='SourceProjectsReposWorkspacesListFilesRequest', response_type_name='ListFilesResponse', supports_download=False)

    def ModifyWorkspace(self, request, global_params=None):
        """Applies an ordered sequence of file modification actions to a workspace.
Returns ABORTED if current_snapshot_id in the request does not refer to
the most recent update to the workspace or if the workspace is
simultaneously modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesModifyWorkspaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
        config = self.GetMethodConfig('ModifyWorkspace')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyWorkspace.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='source.projects.repos.workspaces.modifyWorkspace', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=[], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}:modifyWorkspace', request_field='modifyWorkspaceRequest', request_type_name='SourceProjectsReposWorkspacesModifyWorkspaceRequest', response_type_name='Workspace', supports_download=False)

    def RefreshWorkspace(self, request, global_params=None):
        """Brings a workspace up to date by merging in the changes made between its.
baseline and the revision to which its alias currently refers.
FAILED_PRECONDITION is returned if the alias refers to a revision that is
not a descendant of the workspace baseline, or if the workspace has no
baseline. Returns ABORTED when the workspace is simultaneously modified by
another client.

A refresh may involve merging files in the workspace with files in the
current alias revision. If this merge results in conflicts, then the
workspace is in a merge state: the merge_info field of Workspace will be
populated, and conflicting files in the workspace will contain conflict
markers.

      Args:
        request: (SourceProjectsReposWorkspacesRefreshWorkspaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
        config = self.GetMethodConfig('RefreshWorkspace')
        return self._RunMethod(config, request, global_params=global_params)
    RefreshWorkspace.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='source.projects.repos.workspaces.refreshWorkspace', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=[], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}:refreshWorkspace', request_field='refreshWorkspaceRequest', request_type_name='SourceProjectsReposWorkspacesRefreshWorkspaceRequest', response_type_name='Workspace', supports_download=False)

    def ResolveFiles(self, request, global_params=None):
        """Marks files modified as part of a merge as having been resolved. Returns.
ABORTED when the workspace is simultaneously modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesResolveFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
        config = self.GetMethodConfig('ResolveFiles')
        return self._RunMethod(config, request, global_params=global_params)
    ResolveFiles.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='source.projects.repos.workspaces.resolveFiles', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=[], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}:resolveFiles', request_field='resolveFilesRequest', request_type_name='SourceProjectsReposWorkspacesResolveFilesRequest', response_type_name='Workspace', supports_download=False)

    def RevertRefresh(self, request, global_params=None):
        """If a call to RefreshWorkspace results in conflicts, use RevertRefresh to.
restore the workspace to the state it was in before the refresh.  Returns
FAILED_PRECONDITION if not preceded by a call to RefreshWorkspace, or if
there are no unresolved conflicts remaining. Returns ABORTED when the
workspace is simultaneously modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesRevertRefreshRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
        config = self.GetMethodConfig('RevertRefresh')
        return self._RunMethod(config, request, global_params=global_params)
    RevertRefresh.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='source.projects.repos.workspaces.revertRefresh', ordered_params=['projectId', 'repoName', 'name'], path_params=['name', 'projectId', 'repoName'], query_params=[], relative_path='v1/projects/{projectId}/repos/{repoName}/workspaces/{name}:revertRefresh', request_field='revertRefreshRequest', request_type_name='SourceProjectsReposWorkspacesRevertRefreshRequest', response_type_name='Workspace', supports_download=False)