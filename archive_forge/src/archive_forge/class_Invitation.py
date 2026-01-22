from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class Invitation(CompletableGithubObject):
    """
    This class represents repository invitations. The reference can be found here https://docs.github.com/en/rest/reference/repos#invitations
    """

    def _initAttributes(self) -> None:
        self._id: Attribute[int] = NotSet
        self._permissions: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._invitee: Attribute[NamedUser] = NotSet
        self._inviter: Attribute[NamedUser] = NotSet
        self._url: Attribute[str] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._repository: Attribute[Repository] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value})

    @property
    def id(self) -> int:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def permissions(self) -> str:
        self._completeIfNotSet(self._permissions)
        return self._permissions.value

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def invitee(self) -> NamedUser:
        self._completeIfNotSet(self._invitee)
        return self._invitee.value

    @property
    def inviter(self) -> NamedUser:
        self._completeIfNotSet(self._inviter)
        return self._inviter.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def repository(self) -> Repository:
        self._completeIfNotSet(self._repository)
        return self._repository.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'repository' in attributes:
            self._repository = self._makeClassAttribute(github.Repository.Repository, attributes['repository'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'invitee' in attributes:
            self._invitee = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['invitee'])
        if 'inviter' in attributes:
            self._inviter = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['inviter'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'permissions' in attributes:
            self._permissions = self._makeStringAttribute(attributes['permissions'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])