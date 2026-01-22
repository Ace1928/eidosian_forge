from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
class ProjectsReposAliasesService(base_api.BaseApiService):
    """Service class for the projects_repos_aliases resource."""
    _NAME = 'projects_repos_aliases'

    def __init__(self, client):
        super(SourceV1.ProjectsReposAliasesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new alias. It is an ALREADY_EXISTS error if an alias with that.
name and kind already exists.

      Args:
        request: (SourceProjectsReposAliasesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Alias) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='source.projects.repos.aliases.create', ordered_params=['projectId', 'repoName'], path_params=['projectId', 'repoName'], query_params=['repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/aliases', request_field='alias', request_type_name='SourceProjectsReposAliasesCreateRequest', response_type_name='Alias', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the alias with the given name and kind. Kind cannot be ANY.  If.
the alias does not exist, NOT_FOUND is returned.  If the request provides
a revision ID and the alias does not refer to that revision, ABORTED is
returned.

      Args:
        request: (SourceProjectsReposAliasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='source.projects.repos.aliases.delete', ordered_params=['projectId', 'repoName', 'kind', 'name'], path_params=['kind', 'name', 'projectId', 'repoName'], query_params=['repoId_uid', 'revisionId'], relative_path='v1/projects/{projectId}/repos/{repoName}/aliases/{kind}/{name}', request_field='', request_type_name='SourceProjectsReposAliasesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about an alias. Kind ANY returns a FIXED or.
MOVABLE alias, in that order, and ignores all other kinds.

      Args:
        request: (SourceProjectsReposAliasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Alias) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.aliases.get', ordered_params=['projectId', 'repoName', 'kind', 'name'], path_params=['kind', 'name', 'projectId', 'repoName'], query_params=['repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/aliases/{kind}/{name}', request_field='', request_type_name='SourceProjectsReposAliasesGetRequest', response_type_name='Alias', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of aliases of the given kind. Kind ANY returns all aliases.
in the repo. The order in which the aliases are returned is undefined.

      Args:
        request: (SourceProjectsReposAliasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAliasesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.aliases.list', ordered_params=['projectId', 'repoName'], path_params=['projectId', 'repoName'], query_params=['kind', 'pageSize', 'pageToken', 'repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/aliases', request_field='', request_type_name='SourceProjectsReposAliasesListRequest', response_type_name='ListAliasesResponse', supports_download=False)

    def ListFiles(self, request, global_params=None):
        """ListFiles returns a list of all files in a SourceContext. The.
information about each file includes its path and its hash.
The result is ordered by path. Pagination is supported.

      Args:
        request: (SourceProjectsReposAliasesListFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFilesResponse) The response message.
      """
        config = self.GetMethodConfig('ListFiles')
        return self._RunMethod(config, request, global_params=global_params)
    ListFiles.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.aliases.listFiles', ordered_params=['projectId', 'repoName', 'kind', 'name'], path_params=['kind', 'name', 'projectId', 'repoName'], query_params=['pageSize', 'pageToken', 'sourceContext_cloudRepo_aliasName', 'sourceContext_cloudRepo_repoId_uid', 'sourceContext_cloudRepo_revisionId', 'sourceContext_cloudWorkspace_snapshotId', 'sourceContext_cloudWorkspace_workspaceId_name', 'sourceContext_cloudWorkspace_workspaceId_repoId_projectRepoId_projectId', 'sourceContext_cloudWorkspace_workspaceId_repoId_projectRepoId_repoName', 'sourceContext_cloudWorkspace_workspaceId_repoId_uid', 'sourceContext_gerrit_aliasContext_kind', 'sourceContext_gerrit_aliasContext_name', 'sourceContext_gerrit_aliasName', 'sourceContext_gerrit_gerritProject', 'sourceContext_gerrit_hostUri', 'sourceContext_gerrit_revisionId', 'sourceContext_git_revisionId', 'sourceContext_git_url'], relative_path='v1/projects/{projectId}/repos/{repoName}/aliases/{kind}/{name}:listFiles', request_field='', request_type_name='SourceProjectsReposAliasesListFilesRequest', response_type_name='ListFilesResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the alias with the given name and kind. Kind cannot be ANY.  If.
the alias does not exist, NOT_FOUND is returned. If the request provides
an old revision ID and the alias does not refer to that revision, ABORTED
is returned.

      Args:
        request: (SourceProjectsReposAliasesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Alias) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='source.projects.repos.aliases.update', ordered_params=['projectId', 'repoName', 'aliasesId'], path_params=['aliasesId', 'projectId', 'repoName'], query_params=['oldRevisionId', 'repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/aliases/{aliasesId}', request_field='alias', request_type_name='SourceProjectsReposAliasesUpdateRequest', response_type_name='Alias', supports_download=False)