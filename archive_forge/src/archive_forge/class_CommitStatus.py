from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class CommitStatus(NonCompletableGithubObject):
    """
    This class represents CommitStatuses.The reference can be found here https://docs.github.com/en/rest/reference/repos#statuses
    """

    def _initAttributes(self) -> None:
        self._created_at: Attribute[datetime] = NotSet
        self._creator: Attribute[github.NamedUser.NamedUser] = NotSet
        self._description: Attribute[str] = NotSet
        self._id: Attribute[int] = NotSet
        self._state: Attribute[str] = NotSet
        self._context: Attribute[str] = NotSet
        self._target_url: Attribute[str] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value, 'state': self._state.value, 'context': self._context.value})

    @property
    def created_at(self) -> datetime:
        return self._created_at.value

    @property
    def creator(self) -> github.NamedUser.NamedUser:
        return self._creator.value

    @property
    def description(self) -> str:
        return self._description.value

    @property
    def id(self) -> int:
        return self._id.value

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def context(self) -> str:
        return self._context.value

    @property
    def target_url(self) -> str:
        return self._target_url.value

    @property
    def updated_at(self) -> datetime:
        return self._updated_at.value

    @property
    def url(self) -> str:
        return self._url.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'creator' in attributes:
            self._creator = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['creator'])
        if 'description' in attributes:
            self._description = self._makeStringAttribute(attributes['description'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'state' in attributes:
            self._state = self._makeStringAttribute(attributes['state'])
        if 'context' in attributes:
            self._context = self._makeStringAttribute(attributes['context'])
        if 'target_url' in attributes:
            self._target_url = self._makeStringAttribute(attributes['target_url'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])