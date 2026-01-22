from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def annotation_level(self) -> str:
    return self._annotation_level.value