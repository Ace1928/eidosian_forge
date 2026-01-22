from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GitAuthor
import github.GithubObject
import github.GitObject
import github.GitTreeElement
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class GitTag(CompletableGithubObject):
    """
    This class represents GitTags. The reference can be found here https://docs.github.com/en/rest/reference/git#tags
    """

    def _initAttributes(self) -> None:
        self._message: Attribute[str] = NotSet
        self._object: Attribute[GitObject] = NotSet
        self._sha: Attribute[str] = NotSet
        self._tag: Attribute[str] = NotSet
        self._tagger: Attribute[GitAuthor] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'sha': self._sha.value, 'tag': self._tag.value})

    @property
    def message(self) -> str:
        self._completeIfNotSet(self._message)
        return self._message.value

    @property
    def object(self) -> GitObject:
        self._completeIfNotSet(self._object)
        return self._object.value

    @property
    def sha(self) -> str:
        self._completeIfNotSet(self._sha)
        return self._sha.value

    @property
    def tag(self) -> str:
        self._completeIfNotSet(self._tag)
        return self._tag.value

    @property
    def tagger(self) -> GitAuthor:
        self._completeIfNotSet(self._tagger)
        return self._tagger.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'message' in attributes:
            self._message = self._makeStringAttribute(attributes['message'])
        if 'object' in attributes:
            self._object = self._makeClassAttribute(github.GitObject.GitObject, attributes['object'])
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'tag' in attributes:
            self._tag = self._makeStringAttribute(attributes['tag'])
        if 'tagger' in attributes:
            self._tagger = self._makeClassAttribute(github.GitAuthor.GitAuthor, attributes['tagger'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])