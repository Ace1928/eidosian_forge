from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.NamedUser
import github.Organization
import github.PaginatedList
import github.Repository
import github.TeamDiscussion
from github import Consts
from github.GithubException import UnknownObjectException
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
def update_team_repository(self, repo: Repository, permission: str) -> bool:
    """
        :calls: `PUT /orgs/{org}/teams/{team_slug}/repos/{owner}/{repo} <https://docs.github.com/en/rest/reference/teams#check-team-permissions-for-a-repository>`_
        """
    assert isinstance(repo, github.Repository.Repository) or isinstance(repo, str), repo
    assert isinstance(permission, str), permission
    if isinstance(repo, github.Repository.Repository):
        repo_url_param = repo._identity
    else:
        repo_url_param = urllib.parse.quote(repo)
    put_parameters = {'permission': permission}
    status, _, _ = self._requester.requestJson('PUT', f'{self.organization.url}/teams/{self.slug}/repos/{repo_url_param}', input=put_parameters)
    return status == 204