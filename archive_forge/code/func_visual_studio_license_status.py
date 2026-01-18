from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def visual_studio_license_status(self) -> str:
    self._completeIfNotSet(self._visual_studio_license_status)
    return self._visual_studio_license_status.value