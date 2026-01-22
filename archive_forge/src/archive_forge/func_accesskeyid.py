from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def accesskeyid(self) -> str:
    self._completeIfNotSet(self._accesskeyid)
    return self._accesskeyid.value