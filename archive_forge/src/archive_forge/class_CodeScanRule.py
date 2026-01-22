from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class CodeScanRule(NonCompletableGithubObject):
    """
    This class represents Alerts from code scanning.
    The reference can be found here https://docs.github.com/en/rest/reference/code-scanning.
    """

    def _initAttributes(self) -> None:
        self._id: Attribute[str] = NotSet
        self._name: Attribute[str] = NotSet
        self._severity: Attribute[str] = NotSet
        self._security_severity_level: Attribute[str] = NotSet
        self._description: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self.id, 'name': self.name})

    @property
    def id(self) -> str:
        return self._id.value

    @property
    def name(self) -> str:
        return self._name.value

    @property
    def severity(self) -> str:
        return self._severity.value

    @property
    def security_severity_level(self) -> str:
        return self._security_severity_level.value

    @property
    def description(self) -> str:
        return self._description.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'id' in attributes:
            self._id = self._makeStringAttribute(attributes['id'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'severity' in attributes:
            self._severity = self._makeStringAttribute(attributes['severity'])
        if 'security_severity_level' in attributes:
            self._security_severity_level = self._makeStringAttribute(attributes['security_severity_level'])
        if 'description' in attributes:
            self._description = self._makeStringAttribute(attributes['description'])