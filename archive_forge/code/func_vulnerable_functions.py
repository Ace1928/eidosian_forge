from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
from typing_extensions import TypedDict
import github.AdvisoryVulnerabilityPackage
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def vulnerable_functions(self) -> list[str] | None:
    """
        :type: list of string
        """
    return self._vulnerable_functions.value