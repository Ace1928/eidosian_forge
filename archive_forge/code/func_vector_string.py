from decimal import Decimal
from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def vector_string(self) -> str:
    return self._vector_string.value