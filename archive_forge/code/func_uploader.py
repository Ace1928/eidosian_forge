from __future__ import annotations
from datetime import datetime
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def uploader(self) -> github.NamedUser.NamedUser:
    self._completeIfNotSet(self._uploader)
    return self._uploader.value