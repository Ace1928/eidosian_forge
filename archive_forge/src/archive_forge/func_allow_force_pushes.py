from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
import github.Team
from github.GithubObject import Attribute, NotSet, Opt, is_defined
from github.PaginatedList import PaginatedList
@property
def allow_force_pushes(self) -> bool:
    self._completeIfNotSet(self._allow_force_pushes)
    return self._allow_force_pushes.value