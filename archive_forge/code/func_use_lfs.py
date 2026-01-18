from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def use_lfs(self) -> str:
    self._completeIfNotSet(self._use_lfs)
    return self._use_lfs.value