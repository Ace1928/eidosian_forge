from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
import github.GitCommit
import github.PullRequest
import github.WorkflowJob
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList
@property
def workflow_url(self) -> str:
    self._completeIfNotSet(self._workflow_url)
    return self._workflow_url.value