from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.AdvisoryBase
import github.DependabotAlertVulnerability
from github.GithubObject import Attribute, NotSet
@property
def vulnerabilities(self) -> list[DependabotAlertVulnerability]:
    return self._vulnerabilities.value