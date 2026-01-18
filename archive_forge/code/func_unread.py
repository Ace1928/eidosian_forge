from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Issue
import github.NotificationSubject
import github.PullRequest
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def unread(self) -> bool:
    self._completeIfNotSet(self._unread)
    return self._unread.value