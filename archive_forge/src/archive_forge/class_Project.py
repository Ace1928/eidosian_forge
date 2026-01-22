from typing import (
import requests
from gitlab import cli, client
from gitlab import exceptions as exc
from gitlab import types, utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .access_requests import ProjectAccessRequestManager  # noqa: F401
from .artifacts import ProjectArtifactManager  # noqa: F401
from .audit_events import ProjectAuditEventManager  # noqa: F401
from .badges import ProjectBadgeManager  # noqa: F401
from .boards import ProjectBoardManager  # noqa: F401
from .branches import ProjectBranchManager, ProjectProtectedBranchManager  # noqa: F401
from .ci_lint import ProjectCiLintManager  # noqa: F401
from .clusters import ProjectClusterManager  # noqa: F401
from .commits import ProjectCommitManager  # noqa: F401
from .container_registry import ProjectRegistryRepositoryManager  # noqa: F401
from .custom_attributes import ProjectCustomAttributeManager  # noqa: F401
from .deploy_keys import ProjectKeyManager  # noqa: F401
from .deploy_tokens import ProjectDeployTokenManager  # noqa: F401
from .deployments import ProjectDeploymentManager  # noqa: F401
from .environments import (  # noqa: F401
from .events import ProjectEventManager  # noqa: F401
from .export_import import ProjectExportManager, ProjectImportManager  # noqa: F401
from .files import ProjectFileManager  # noqa: F401
from .hooks import ProjectHookManager  # noqa: F401
from .integrations import ProjectIntegrationManager, ProjectServiceManager  # noqa: F401
from .invitations import ProjectInvitationManager  # noqa: F401
from .issues import ProjectIssueManager  # noqa: F401
from .iterations import ProjectIterationManager  # noqa: F401
from .job_token_scope import ProjectJobTokenScopeManager  # noqa: F401
from .jobs import ProjectJobManager  # noqa: F401
from .labels import ProjectLabelManager  # noqa: F401
from .members import ProjectMemberAllManager, ProjectMemberManager  # noqa: F401
from .merge_request_approvals import (  # noqa: F401
from .merge_requests import ProjectMergeRequestManager  # noqa: F401
from .merge_trains import ProjectMergeTrainManager  # noqa: F401
from .milestones import ProjectMilestoneManager  # noqa: F401
from .notes import ProjectNoteManager  # noqa: F401
from .notification_settings import ProjectNotificationSettingsManager  # noqa: F401
from .packages import GenericPackageManager, ProjectPackageManager  # noqa: F401
from .pages import ProjectPagesDomainManager  # noqa: F401
from .pipelines import (  # noqa: F401
from .project_access_tokens import ProjectAccessTokenManager  # noqa: F401
from .push_rules import ProjectPushRulesManager  # noqa: F401
from .releases import ProjectReleaseManager  # noqa: F401
from .repositories import RepositoryMixin
from .resource_groups import ProjectResourceGroupManager
from .runners import ProjectRunnerManager  # noqa: F401
from .secure_files import ProjectSecureFileManager  # noqa: F401
from .snippets import ProjectSnippetManager  # noqa: F401
from .statistics import (  # noqa: F401
from .tags import ProjectProtectedTagManager, ProjectTagManager  # noqa: F401
from .triggers import ProjectTriggerManager  # noqa: F401
from .users import ProjectUserManager  # noqa: F401
from .variables import ProjectVariableManager  # noqa: F401
from .wikis import ProjectWikiManager  # noqa: F401
class Project(RefreshMixin, SaveMixin, ObjectDeleteMixin, RepositoryMixin, UploadMixin, RESTObject):
    _repr_attr = 'path_with_namespace'
    _upload_path = '/projects/{id}/uploads'
    access_tokens: ProjectAccessTokenManager
    accessrequests: ProjectAccessRequestManager
    additionalstatistics: ProjectAdditionalStatisticsManager
    approvalrules: ProjectApprovalRuleManager
    approvals: ProjectApprovalManager
    artifacts: ProjectArtifactManager
    audit_events: ProjectAuditEventManager
    badges: ProjectBadgeManager
    boards: ProjectBoardManager
    branches: ProjectBranchManager
    ci_lint: ProjectCiLintManager
    clusters: ProjectClusterManager
    commits: ProjectCommitManager
    customattributes: ProjectCustomAttributeManager
    deployments: ProjectDeploymentManager
    deploytokens: ProjectDeployTokenManager
    environments: ProjectEnvironmentManager
    events: ProjectEventManager
    exports: ProjectExportManager
    files: ProjectFileManager
    forks: 'ProjectForkManager'
    generic_packages: GenericPackageManager
    groups: ProjectGroupManager
    hooks: ProjectHookManager
    imports: ProjectImportManager
    integrations: ProjectIntegrationManager
    invitations: ProjectInvitationManager
    issues: ProjectIssueManager
    issues_statistics: ProjectIssuesStatisticsManager
    iterations: ProjectIterationManager
    jobs: ProjectJobManager
    job_token_scope: ProjectJobTokenScopeManager
    keys: ProjectKeyManager
    labels: ProjectLabelManager
    members: ProjectMemberManager
    members_all: ProjectMemberAllManager
    mergerequests: ProjectMergeRequestManager
    merge_trains: ProjectMergeTrainManager
    milestones: ProjectMilestoneManager
    notes: ProjectNoteManager
    notificationsettings: ProjectNotificationSettingsManager
    packages: ProjectPackageManager
    pagesdomains: ProjectPagesDomainManager
    pipelines: ProjectPipelineManager
    pipelineschedules: ProjectPipelineScheduleManager
    protected_environments: ProjectProtectedEnvironmentManager
    protectedbranches: ProjectProtectedBranchManager
    protectedtags: ProjectProtectedTagManager
    pushrules: ProjectPushRulesManager
    releases: ProjectReleaseManager
    resource_groups: ProjectResourceGroupManager
    remote_mirrors: 'ProjectRemoteMirrorManager'
    repositories: ProjectRegistryRepositoryManager
    runners: ProjectRunnerManager
    secure_files: ProjectSecureFileManager
    services: ProjectServiceManager
    snippets: ProjectSnippetManager
    storage: 'ProjectStorageManager'
    tags: ProjectTagManager
    triggers: ProjectTriggerManager
    users: ProjectUserManager
    variables: ProjectVariableManager
    wikis: ProjectWikiManager

    @cli.register_custom_action('Project', ('forked_from_id',))
    @exc.on_http_error(exc.GitlabCreateError)
    def create_fork_relation(self, forked_from_id: int, **kwargs: Any) -> None:
        """Create a forked from/to relation between existing projects.

        Args:
            forked_from_id: The ID of the project that was forked from
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the relation could not be created
        """
        path = f'/projects/{self.encoded_id}/fork/{forked_from_id}'
        self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabDeleteError)
    def delete_fork_relation(self, **kwargs: Any) -> None:
        """Delete a forked relation between existing projects.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/fork'
        self.manager.gitlab.http_delete(path, **kwargs)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabGetError)
    def languages(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Get languages used in the project with percentage value.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/languages'
        return self.manager.gitlab.http_get(path, **kwargs)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabCreateError)
    def star(self, **kwargs: Any) -> None:
        """Star a project.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/star'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        self._update_attrs(server_data)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabDeleteError)
    def unstar(self, **kwargs: Any) -> None:
        """Unstar a project.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/unstar'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        self._update_attrs(server_data)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabCreateError)
    def archive(self, **kwargs: Any) -> None:
        """Archive a project.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/archive'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        self._update_attrs(server_data)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabDeleteError)
    def unarchive(self, **kwargs: Any) -> None:
        """Unarchive a project.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/unarchive'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        self._update_attrs(server_data)

    @cli.register_custom_action('Project', ('group_id', 'group_access'), ('expires_at',))
    @exc.on_http_error(exc.GitlabCreateError)
    def share(self, group_id: int, group_access: int, expires_at: Optional[str]=None, **kwargs: Any) -> None:
        """Share the project with a group.

        Args:
            group_id: ID of the group.
            group_access: Access level for the group.
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/share'
        data = {'group_id': group_id, 'group_access': group_access, 'expires_at': expires_at}
        self.manager.gitlab.http_post(path, post_data=data, **kwargs)

    @cli.register_custom_action('Project', ('group_id',))
    @exc.on_http_error(exc.GitlabDeleteError)
    def unshare(self, group_id: int, **kwargs: Any) -> None:
        """Delete a shared project link within a group.

        Args:
            group_id: ID of the group.
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/share/{group_id}'
        self.manager.gitlab.http_delete(path, **kwargs)

    @cli.register_custom_action('Project', ('ref', 'token'))
    @exc.on_http_error(exc.GitlabCreateError)
    def trigger_pipeline(self, ref: str, token: str, variables: Optional[Dict[str, Any]]=None, **kwargs: Any) -> ProjectPipeline:
        """Trigger a CI build.

        See https://gitlab.com/help/ci/triggers/README.md#trigger-a-build

        Args:
            ref: Commit to build; can be a branch name or a tag
            token: The trigger token
            variables: Variables passed to the build script
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server failed to perform the request
        """
        variables = variables or {}
        path = f'/projects/{self.encoded_id}/trigger/pipeline'
        post_data = {'ref': ref, 'token': token, 'variables': variables}
        attrs = self.manager.gitlab.http_post(path, post_data=post_data, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(attrs, dict)
        return ProjectPipeline(self.pipelines, attrs)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabHousekeepingError)
    def housekeeping(self, **kwargs: Any) -> None:
        """Start the housekeeping task.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabHousekeepingError: If the server failed to perform the
                                     request
        """
        path = f'/projects/{self.encoded_id}/housekeeping'
        self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabRestoreError)
    def restore(self, **kwargs: Any) -> None:
        """Restore a project marked for deletion.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabRestoreError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/restore'
        self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('Project', optional=('wiki',))
    @exc.on_http_error(exc.GitlabGetError)
    def snapshot(self, wiki: bool=False, streamed: bool=False, action: Optional[Callable[[bytes], None]]=None, chunk_size: int=1024, *, iterator: bool=False, **kwargs: Any) -> Optional[Union[bytes, Iterator[Any]]]:
        """Return a snapshot of the repository.

        Args:
            wiki: If True return the wiki repository
            streamed: If True the data will be processed by chunks of
                `chunk_size` and each chunk is passed to `action` for
                treatment.
            iterator: If True directly return the underlying response
                iterator
            action: Callable responsible of dealing with chunk of
                data
            chunk_size: Size of each chunk
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the content could not be retrieved

        Returns:
            The uncompressed tar archive of the repository
        """
        path = f'/projects/{self.encoded_id}/snapshot'
        result = self.manager.gitlab.http_get(path, streamed=streamed, raw=True, wiki=wiki, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, requests.Response)
        return utils.response_content(result, streamed, action, chunk_size, iterator=iterator)

    @cli.register_custom_action('Project', ('scope', 'search'))
    @exc.on_http_error(exc.GitlabSearchError)
    def search(self, scope: str, search: str, **kwargs: Any) -> Union[client.GitlabList, List[Dict[str, Any]]]:
        """Search the project resources matching the provided string.'

        Args:
            scope: Scope of the search
            search: Search string
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabSearchError: If the server failed to perform the request

        Returns:
            A list of dicts describing the resources found.
        """
        data = {'scope': scope, 'search': search}
        path = f'/projects/{self.encoded_id}/search'
        return self.manager.gitlab.http_list(path, query_data=data, **kwargs)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabCreateError)
    def mirror_pull(self, **kwargs: Any) -> None:
        """Start the pull mirroring process for the project.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server failed to perform the request
        """
        path = f'/projects/{self.encoded_id}/mirror/pull'
        self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('Project')
    @exc.on_http_error(exc.GitlabGetError)
    def mirror_pull_details(self, **kwargs: Any) -> Dict[str, Any]:
        """Get a project's pull mirror details.

        Introduced in GitLab 15.5.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            dict of the parsed json returned by the server
        """
        path = f'/projects/{self.encoded_id}/mirror/pull'
        result = self.manager.gitlab.http_get(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, dict)
        return result

    @cli.register_custom_action('Project', ('to_namespace',))
    @exc.on_http_error(exc.GitlabTransferProjectError)
    def transfer(self, to_namespace: Union[int, str], **kwargs: Any) -> None:
        """Transfer a project to the given namespace ID

        Args:
            to_namespace: ID or path of the namespace to transfer the
            project to
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTransferProjectError: If the project could not be transferred
        """
        path = f'/projects/{self.encoded_id}/transfer'
        self.manager.gitlab.http_put(path, post_data={'namespace': to_namespace}, **kwargs)