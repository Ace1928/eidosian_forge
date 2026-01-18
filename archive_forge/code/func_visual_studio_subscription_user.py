from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def visual_studio_subscription_user(self) -> bool:
    self._completeIfNotSet(self._visual_studio_subscription_user)
    return self._visual_studio_subscription_user.value