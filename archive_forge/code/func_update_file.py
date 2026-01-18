from __future__ import annotations
import collections
import urllib.parse
from base64 import b64encode
from collections.abc import Iterable
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.AdvisoryCredit
import github.AdvisoryVulnerability
import github.Artifact
import github.AuthenticatedUser
import github.Autolink
import github.Branch
import github.CheckRun
import github.CheckSuite
import github.Clones
import github.CodeScanAlert
import github.Commit
import github.CommitComment
import github.Comparison
import github.ContentFile
import github.DependabotAlert
import github.Deployment
import github.Download
import github.Environment
import github.EnvironmentDeploymentBranchPolicy
import github.EnvironmentProtectionRule
import github.EnvironmentProtectionRuleReviewer
import github.Event
import github.GitBlob
import github.GitCommit
import github.GithubObject
import github.GitRef
import github.GitRelease
import github.GitReleaseAsset
import github.GitTag
import github.GitTree
import github.Hook
import github.HookDelivery
import github.Invitation
import github.Issue
import github.IssueComment
import github.IssueEvent
import github.Label
import github.License
import github.Milestone
import github.NamedUser
import github.Notification
import github.Organization
import github.PaginatedList
import github.Path
import github.Permissions
import github.Project
import github.PublicKey
import github.PullRequest
import github.PullRequestComment
import github.Referrer
import github.RepositoryAdvisory
import github.RepositoryKey
import github.RepositoryPreferences
import github.Secret
import github.SelfHostedActionsRunner
import github.SourceImport
import github.Stargazer
import github.StatsCodeFrequency
import github.StatsCommitActivity
import github.StatsContributor
import github.StatsParticipation
import github.StatsPunchCard
import github.Tag
import github.Team
import github.Variable
import github.View
import github.Workflow
import github.WorkflowRun
from github import Consts
from github.Environment import Environment
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def update_file(self, path: str, message: str, content: bytes | str, sha: str, branch: Opt[str]=NotSet, committer: Opt[InputGitAuthor]=NotSet, author: Opt[InputGitAuthor]=NotSet) -> dict[str, ContentFile | Commit]:
    """This method updates a file in a repository

        :calls: `PUT /repos/{owner}/{repo}/contents/{path} <https://docs.github.com/en/rest/reference/repos#create-or-update-file-contents>`_
        :param path: string, Required. The content path.
        :param message: string, Required. The commit message.
        :param content: string, Required. The updated file content, either base64 encoded, or ready to be encoded.
        :param sha: string, Required. The blob SHA of the file being replaced.
        :param branch: string. The branch name. Default: the repository’s default branch (usually master)
        :param committer: InputGitAuthor, (optional), if no information is given the authenticated user's information will be used. You must specify both a name and email.
        :param author: InputGitAuthor, (optional), if omitted this will be filled in with committer information. If passed, you must specify both a name and email.
        :rtype: {
            'content': :class:`ContentFile <github.ContentFile.ContentFile>`:,
            'commit': :class:`Commit <github.Commit.Commit>`}
        """
    assert isinstance(path, str)
    assert isinstance(message, str)
    assert isinstance(content, (str, bytes))
    assert isinstance(sha, str)
    assert is_optional(branch, str)
    assert is_optional(author, github.InputGitAuthor)
    assert is_optional(committer, github.InputGitAuthor)
    if not isinstance(content, bytes):
        content = content.encode('utf-8')
    content = b64encode(content).decode('utf-8')
    put_parameters: dict[str, Any] = {'message': message, 'content': content, 'sha': sha}
    if is_defined(branch):
        put_parameters['branch'] = branch
    if is_defined(author):
        put_parameters['author'] = author._identity
    if is_defined(committer):
        put_parameters['committer'] = committer._identity
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/contents/{urllib.parse.quote(path)}', input=put_parameters)
    return {'commit': github.Commit.Commit(self._requester, headers, data['commit'], completed=True), 'content': github.ContentFile.ContentFile(self._requester, headers, data['content'], completed=False)}