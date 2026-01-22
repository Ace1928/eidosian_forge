from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class InstallationAuthorization(NonCompletableGithubObject):
    """
    This class represents InstallationAuthorizations
    """

    def _initAttributes(self) -> None:
        self._token: Attribute[str] = NotSet
        self._expires_at: Attribute[datetime] = NotSet
        self._on_behalf_of: Attribute[NamedUser] = NotSet
        self._permissions: Attribute[dict] = NotSet
        self._repository_selection: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'expires_at': self._expires_at.value})

    @property
    def token(self) -> str:
        return self._token.value

    @property
    def expires_at(self) -> datetime:
        return self._expires_at.value

    @property
    def on_behalf_of(self) -> NamedUser:
        return self._on_behalf_of.value

    @property
    def permissions(self) -> dict:
        return self._permissions.value

    @property
    def repository_selection(self) -> str:
        return self._repository_selection.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'token' in attributes:
            self._token = self._makeStringAttribute(attributes['token'])
        if 'expires_at' in attributes:
            self._expires_at = self._makeDatetimeAttribute(attributes['expires_at'])
        if 'on_behalf_of' in attributes:
            self._on_behalf_of = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['on_behalf_of'])
        if 'permissions' in attributes:
            self._permissions = self._makeDictAttribute(attributes['permissions'])
        if 'repository_selection' in attributes:
            self._repository_selection = self._makeStringAttribute(attributes['repository_selection'])