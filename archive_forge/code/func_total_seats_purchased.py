from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.NamedEnterpriseUser import NamedEnterpriseUser
from github.PaginatedList import PaginatedList
@property
def total_seats_purchased(self) -> int:
    return self._total_seats_purchased.value