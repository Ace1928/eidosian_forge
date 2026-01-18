from __future__ import annotations
from datetime import datetime
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
def update_asset(self, name: str, label: str='') -> GitReleaseAsset:
    """
        Update asset metadata.
        """
    assert isinstance(name, str), name
    assert isinstance(label, str), label
    post_parameters = {'name': name, 'label': label}
    headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
    return GitReleaseAsset(self._requester, headers, data, completed=True)