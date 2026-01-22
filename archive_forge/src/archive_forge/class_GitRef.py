from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.GitObject
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
class GitRef(CompletableGithubObject):
    """
    This class represents GitRefs. The reference can be found here https://docs.github.com/en/rest/reference/git#references
    """

    def _initAttributes(self) -> None:
        self._object: Attribute[GitObject] = NotSet
        self._ref: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'ref': self._ref.value})

    @property
    def object(self) -> GitObject:
        self._completeIfNotSet(self._object)
        return self._object.value

    @property
    def ref(self) -> str:
        self._completeIfNotSet(self._ref)
        return self._ref.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    def delete(self) -> None:
        """
        :calls: `DELETE /repos/{owner}/{repo}/git/refs/{ref} <https://docs.github.com/en/rest/reference/git#references>`_
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)

    def edit(self, sha: str, force: Opt[bool]=NotSet) -> None:
        """
        :calls: `PATCH /repos/{owner}/{repo}/git/refs/{ref} <https://docs.github.com/en/rest/reference/git#references>`_
        """
        assert isinstance(sha, str), sha
        assert is_optional(force, bool), force
        post_parameters = NotSet.remove_unset_items({'sha': sha, 'force': force})
        headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
        self._useAttributes(data)

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'object' in attributes:
            self._object = self._makeClassAttribute(github.GitObject.GitObject, attributes['object'])
        if 'ref' in attributes:
            self._ref = self._makeStringAttribute(attributes['ref'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])