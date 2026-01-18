from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Consts
import github.DeploymentStatus
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
@property
def transient_environment(self) -> bool:
    self._completeIfNotSet(self._transient_environment)
    return self._transient_environment.value