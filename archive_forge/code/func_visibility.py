from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NotSet
from github.PaginatedList import PaginatedList
from github.Repository import Repository
from github.Secret import Secret
@property
def visibility(self) -> str:
    """
        :type: string
        """
    self._completeIfNotSet(self._visibility)
    return self._visibility.value