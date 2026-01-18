from __future__ import annotations
from datetime import datetime
from typing import Any
from github.CVSS import CVSS
from github.CWE import CWE
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def withdrawn_at(self) -> datetime:
    return self._withdrawn_at.value