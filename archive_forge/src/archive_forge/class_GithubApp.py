from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class GithubApp(CompletableGithubObject):
    """
    This class represents github apps. The reference can be found here https://docs.github.com/en/rest/reference/apps
    """

    def _initAttributes(self) -> None:
        self._created_at: Attribute[datetime] = NotSet
        self._description: Attribute[str] = NotSet
        self._events: Attribute[list[str]] = NotSet
        self._external_url: Attribute[str] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._id: Attribute[int] = NotSet
        self._name: Attribute[str] = NotSet
        self._owner: Attribute[github.NamedUser.NamedUser] = NotSet
        self._permissions: Attribute[dict[str, str]] = NotSet
        self._slug: Attribute[str] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value, 'url': self._url.value})

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def description(self) -> str:
        self._completeIfNotSet(self._description)
        return self._description.value

    @property
    def events(self) -> list[str]:
        self._completeIfNotSet(self._events)
        return self._events.value

    @property
    def external_url(self) -> str:
        self._completeIfNotSet(self._external_url)
        return self._external_url.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def id(self) -> int:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def name(self) -> str:
        self._completeIfNotSet(self._name)
        return self._name.value

    @property
    def owner(self) -> github.NamedUser.NamedUser:
        self._completeIfNotSet(self._owner)
        return self._owner.value

    @property
    def permissions(self) -> dict[str, str]:
        self._completeIfNotSet(self._permissions)
        return self._permissions.value

    @property
    def slug(self) -> str:
        self._completeIfNotSet(self._slug)
        return self._slug.value

    @property
    def updated_at(self) -> datetime:
        self._completeIfNotSet(self._updated_at)
        return self._updated_at.value

    @property
    def url(self) -> str:
        return self._url.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'description' in attributes:
            self._description = self._makeStringAttribute(attributes['description'])
        if 'events' in attributes:
            self._events = self._makeListOfStringsAttribute(attributes['events'])
        if 'external_url' in attributes:
            self._external_url = self._makeStringAttribute(attributes['external_url'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'owner' in attributes:
            self._owner = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['owner'])
        if 'permissions' in attributes:
            self._permissions = self._makeDictAttribute(attributes['permissions'])
        if 'slug' in attributes:
            self._slug = self._makeStringAttribute(attributes['slug'])
            self._url = self._makeStringAttribute(f'/apps/{attributes['slug']}')
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])