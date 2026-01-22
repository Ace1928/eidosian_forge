from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def acl(self) -> str:
    self._completeIfNotSet(self._acl)
    return self._acl.value