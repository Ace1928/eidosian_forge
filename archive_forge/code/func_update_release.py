from __future__ import annotations
from datetime import datetime
from os.path import basename
from typing import Any, BinaryIO
import github.GitReleaseAsset
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts
def update_release(self, name: str, message: str, draft: bool=False, prerelease: bool=False, tag_name: Opt[str]=NotSet, target_commitish: Opt[str]=NotSet) -> GitRelease:
    """
        :calls: `PATCH /repos/{owner}/{repo}/releases/{release_id} <https://docs.github.com/en/rest/reference/repos#update-a-release>`_
        """
    assert tag_name is NotSet or isinstance(tag_name, str), 'tag_name must be a str/unicode object'
    assert target_commitish is NotSet or isinstance(target_commitish, str), 'target_commitish must be a str/unicode object'
    assert isinstance(name, str), name
    assert isinstance(message, str), message
    assert isinstance(draft, bool), draft
    assert isinstance(prerelease, bool), prerelease
    if tag_name is NotSet:
        tag_name = self.tag_name
    post_parameters = {'tag_name': tag_name, 'name': name, 'body': message, 'draft': draft, 'prerelease': prerelease}
    if target_commitish is not NotSet:
        post_parameters['target_commitish'] = target_commitish
    headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
    return github.GitRelease.GitRelease(self._requester, headers, data, completed=True)