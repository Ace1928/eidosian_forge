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
def add_to_repos(self, repo: Repository) -> None:
    """
        :calls: `PUT /teams/{id}/repos/{org}/{repo} <https://docs.github.com/en/rest/reference/teams>`_
        """
    assert isinstance(repo, github.Repository.Repository), repo
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/repos/{repo._identity}')