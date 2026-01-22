from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.GitObject
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
def _useAttributes(self, attributes: dict[str, Any]) -> None:
    if 'object' in attributes:
        self._object = self._makeClassAttribute(github.GitObject.GitObject, attributes['object'])
    if 'ref' in attributes:
        self._ref = self._makeStringAttribute(attributes['ref'])
    if 'url' in attributes:
        self._url = self._makeStringAttribute(attributes['url'])