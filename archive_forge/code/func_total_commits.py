from __future__ import annotations
from typing import Any
import github.Commit
import github.File
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
@property
def total_commits(self) -> int:
    self._completeIfNotSet(self._total_commits)
    return self._total_commits.value