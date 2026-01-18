from datetime import datetime
from typing import Any, Dict
import github.GithubObject
from github.GithubObject import Attribute
@property
def week(self) -> datetime:
    return self._week.value