from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def variables_url(self) -> str:
    """
        :type: string
        """
    return self._variables_url.value