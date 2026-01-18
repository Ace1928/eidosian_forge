from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def vcs_url(self) -> str:
    self._completeIfNotSet(self._vcs_url)
    return self._vcs_url.value